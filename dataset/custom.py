import os.path as osp
from abc import ABCMeta, abstractmethod
import warnings
from collections import OrderedDict

import cv2
import numpy as np
import albumentations as A
from torch.utils.data import Dataset

import sys
import os
from pathlib import Path

FILE = Path(__file__).resolve()
ROOT = FILE.parents[1]  # root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative
RANK = int(os.getenv('RANK', -1))

from core.fileio import HardDiskBackend, list_from_file


class CustomDataset(Dataset):
    """Custom dataset for semantic segmentation. An example of file structure
    is as followed.语义分割自定义数据集。文件结构示例如下：

    .. code-block:: none

        ├── data
        │   ├── my_dataset
        │   │   ├── img_dir
        │   │   │   ├── train
        │   │   │   │   ├── xxx{img_suffix}
        │   │   │   │   ├── yyy{img_suffix}
        │   │   │   │   ├── zzz{img_suffix}
        │   │   │   ├── val
        │   │   ├── ann_dir
        │   │   │   ├── train
        │   │   │   │   ├── xxx{seg_map_suffix}
        │   │   │   │   ├── yyy{seg_map_suffix}
        │   │   │   │   ├── zzz{seg_map_suffix}
        │   │   │   ├── val

    CustomDataset的图像/语义标注图文件对应当保持相同前缀（仅后缀不同）。有效的文件名对应关系应为
    ``xxx{图片后缀}`` 和 ``xxx{标注图后缀}``（扩展名包含在后缀中）。若提供split参数，
    则 ``xxx`` 由txt文件指定；否则将加载 ``img_dir/`` 和 ``ann_dir`` 下的所有文件。


    Args:
        pipeline (str): albumentations 数据增强yaml配置文件路径
        img_dir (str): 图像目录路径
        img_suffix (str): 图像文件后缀。默认：'.jpg'
        ann_dir (str, 可选): 标注文件目录路径。默认：None
        seg_map_suffix (str): 语义分割图后缀。默认：'.png'
        split (str, 可选): 数据集划分文件。若指定，则仅加载split文件中列出的文件；
            否则加载img_dir/ann_dir下所有文件。默认：None
        data_root (str, 可选): img_dir/ann_dir的根目录。默认：None
        test_mode (bool): 测试模式开关。若为True则不加载标注。默认：False
        ignore_index (int): 需忽略的标签索引值。默认：255
        reduce_zero_label (bool): 是否将0号标签标记为忽略。默认：False
        classes (str | Sequence[str], 可选): 指定加载的类别名称。若为None则使用``cls.CLASSES``。默认：None
        palette (Sequence[Sequence[int]]] | np.ndarray | None):
            分割图的调色板。若为None且self.PALETTE为None，则生成随机调色板。默认：None
        file_client (class): 文件客户端
    """

    CLASSES = None
    PALETTE = None

    def __init__(self,
                 pipeline,
                 img_dir,
                 img_suffix='.jpg',
                 ann_dir=None,
                 seg_map_suffix='.png',
                 split=None,
                 data_root=None,
                 test_mode=False,
                 ignore_index=255,
                 reduce_zero_label=False,
                 classes=None,
                 palette=None,
                 ori_img_size=None,  # (720, 1280)
                 return_ori_seg_gt=False,
                 file_client=HardDiskBackend):
        self.pipeline = A.load(filepath_or_buffer=pipeline, data_format='yaml')
        self.img_dir = img_dir
        self.img_suffix = img_suffix
        self.ann_dir = ann_dir
        self.seg_map_suffix = seg_map_suffix
        self.split = split
        self.data_root = data_root
        self.test_mode = test_mode
        self.ignore_index = ignore_index
        self.reduce_zero_label = reduce_zero_label
        self.label_map = None
        self.CLASSES, self.PALETTE = self.get_classes_and_palette(classes, palette)

        self.ori_img_size = ori_img_size
        if self.ori_img_size:
            assert isinstance(ori_img_size, tuple) and len(ori_img_size) == 2
            for value in ori_img_size:
                assert isinstance(value, int)

        self.file_client = file_client
        self.return_ori_seg_gt = return_ori_seg_gt  # 是否在data_infos中返回原始标签变量
        if test_mode:
            assert self.CLASSES is not None, \
                '`cls.CLASSES` or `classes` should be specified when testing'

        # join paths if data_root is specified
        if self.data_root is not None:
            if not osp.isabs(self.img_dir):
                self.img_dir = osp.join(self.data_root, self.img_dir)
            if not (self.ann_dir is None or osp.isabs(self.ann_dir)):
                self.ann_dir = osp.join(self.data_root, self.ann_dir)
            if not (self.split is None or osp.isabs(self.split)):
                self.split = osp.join(self.data_root, self.split)

        # load annotations
        self.img_infos = self.load_annotations(self.img_dir, self.img_suffix,
                                               self.ann_dir,
                                               self.seg_map_suffix, self.split)

    def __len__(self):
        """Total number of samples of data."""
        return len(self.img_infos)

    def load_annotations(self, img_dir, img_suffix, ann_dir, seg_map_suffix,
                         split):
        """Load annotation from directory.

        Args:
            img_dir (str): Path to image directory
            img_suffix (str): Suffix of images.
            ann_dir (str|None): Path to annotation directory.
            seg_map_suffix (str|None): Suffix of segmentation maps.
            split (str|None): Split txt file. If split is specified, only file
                with suffix in the splits will be loaded. Otherwise, all images
                in img_dir/ann_dir will be loaded. Default: None

        Returns:
            list[dict]: All image info of dataset.
        """

        img_infos = []
        if split is not None:
            lines = list_from_file(split)
            for line in lines:
                img_name = line.strip()
                img_info = dict(filename=img_name + img_suffix)
                if ann_dir is not None:
                    seg_map = img_name + seg_map_suffix
                    img_info.update(dict(ann_filename=seg_map))
                img_infos.append(img_info)
        else:
            for img in self.file_client.list_dir_or_file(dir_path=img_dir,
                                                         list_dir=False,
                                                         suffix=img_suffix,
                                                         recursive=True):
                img_info = dict(filename=img)
                if ann_dir is not None:
                    seg_map = img.replace(img_suffix, seg_map_suffix)
                    img_info.update(dict(ann_filename=seg_map))
                img_infos.append(img_info)
            img_infos = sorted(img_infos, key=lambda x: x['filename'])

        print(f'Loaded {len(img_infos)} images')
        return img_infos

    def get_ann_info(self, idx):
        """Get annotation by index.

        Args:
            idx (int): Index of data.

        Returns:
            dict: Annotation info of specified index.
        """

        return self.img_infos[idx]['ann']

    def prepare_data_info(self, idx):
        img_info = self.img_infos[idx]
        data_infos = dict(img_file_path=osp.join(self.img_dir,
                                                 img_info['filename']),
                          ann_file_path=osp.join(self.ann_dir,
                                                 img_info['ann_filename']))
        if not self.ori_img_size:
            size = cv2.imread(data_infos['img_file_path']).shape[:2]
            data_infos.update(dict(ori_img_size_hw=size))

        data_infos.update(dict(return_ori_seg_gt=self.return_ori_seg_gt))
        return data_infos

    def __getitem__(self, idx):
        """Get training/test data after pipeline.

        Args:
            idx (int): Index of data.

        Returns:
            dict: Training/test data (with annotation if `test_mode` is set
                False).
        """
        infos = self.prepare_data_info(idx)

        if self.test_mode:
            return self.prepare_test__data(infos)
        else:
            return self.prepare_train_val_data(infos)

    @abstractmethod
    def prepare_train_val_data(self, infos):
        """Placeholder for Forward function for training."""
        pass

    @abstractmethod
    def prepare_test_data(self, infos):
        """Placeholder for Forward function for training."""
        pass

    def format_results(self, results, imgfile_prefix, indices=None, **kwargs):
        """Place holder to format result to dataset specific output."""
        raise NotImplementedError

    def get_gt_seg_map_by_idx(self, index):
        """Get one ground truth segmentation map for evaluation."""
        ann_info = self.get_ann_info(index)
        results = dict(ann_info=ann_info)
        self.pre_pipeline(results)
        self.gt_seg_map_loader(results)
        return results['gt_semantic_seg']

    def get_gt_seg_maps(self, efficient_test=None):
        """Get ground truth segmentation maps for evaluation."""
        if efficient_test is not None:
            warnings.warn(
                'DeprecationWarning: ``efficient_test`` has been deprecated '
                'since MMSeg v0.16, the ``get_gt_seg_maps()`` is CPU memory '
                'friendly by default. ')

        for idx in range(len(self)):
            ann_info = self.get_ann_info(idx)
            results = dict(ann_info=ann_info)
            self.pre_pipeline(results)
            self.gt_seg_map_loader(results)
            yield results['gt_semantic_seg']

    def get_classes_and_palette(self, classes=None, palette=None):
        """Get class names of current dataset.

        Args:
            classes (Sequence[str] | str | None): If classes is None, use
                default CLASSES defined by builtin dataset. If classes is a
                string, take it as a file name. The file contains the name of
                classes where each line contains one class name. If classes is
                a tuple or list, override the CLASSES defined by the dataset.
            palette (Sequence[Sequence[int]]] | np.ndarray | None):
                The palette of segmentation map. If None is given, random
                palette will be generated. Default: None
        """
        if classes is None:
            self.custom_classes = False
            return self.CLASSES, self.PALETTE

        self.custom_classes = True
        if isinstance(classes, str):
            # take it as a file path
            class_names = list_from_file(classes)
        elif isinstance(classes, (tuple, list)):
            class_names = classes
        else:
            raise ValueError(f'Unsupported type {type(classes)} of classes.')

        if self.CLASSES:
            if not set(class_names).issubset(self.CLASSES):
                raise ValueError('classes is not a subset of CLASSES.')

            # dictionary, its keys are the old label ids and its values
            # are the new label ids.
            # used for changing pixel labels in load_annotations.
            self.label_map = {}
            for i, c in enumerate(self.CLASSES):
                if c not in class_names:
                    self.label_map[i] = -1
                else:
                    self.label_map[i] = class_names.index(c)

        palette = self.get_palette_for_custom_classes(class_names, palette)

        return class_names, palette

    def get_palette_for_custom_classes(self, class_names, palette=None):

        if self.label_map is not None:
            # return subset of palette
            palette = []
            for old_id, new_id in sorted(
                    self.label_map.items(), key=lambda x: x[1]):
                if new_id != -1:
                    palette.append(self.PALETTE[old_id])
            palette = type(self.PALETTE)(palette)

        elif palette is None:
            if self.PALETTE is None:
                # Get random state before set seed, and restore
                # random state later.
                # It will prevent loss of randomness, as the palette
                # may be different in each iteration if not specified.
                # See: https://github.com/open-mmlab/mmdetection/issues/5844
                state = np.random.get_state()
                np.random.seed(42)
                # random palette
                palette = np.random.randint(0, 255, size=(len(class_names), 3))
                np.random.set_state(state)
            else:
                palette = self.PALETTE

        return palette
