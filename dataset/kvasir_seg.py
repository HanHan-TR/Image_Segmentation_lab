import cv2 as cv
import torch
import sys
import os
from pathlib import Path

FILE = Path(__file__).resolve()
ROOT = FILE.parents[1]  # root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative
RANK = int(os.getenv('RANK', -1))

from dataset.custom import CustomDataset
from core.registry import DATASET


@DATASET.register()
class KvasirSegDataset(CustomDataset):
    """
    """

    CLASSES = ['background', 'polyp']
    PALETTE = [[0, 0, 0], [0, 63, 255]]

    def __init__(self, **kwargs):
        super(KvasirSegDataset, self).__init__(**kwargs)

    def prepare_train_val_data(self, infos):
        im = cv.imread(infos['img_file_path'], cv.COLOR_BGR2RGB)
        ori_gt = cv.imread(infos['ann_file_path'], cv.IMREAD_GRAYSCALE)
        ori_gt[ori_gt >= 255] = 1
        ori_gt[ori_gt < 255] = 0

        if infos['return_ori_seg_gt']:
            infos.update(dict(ori_gt=ori_gt))

        # 数据增强
        augmented = self.pipeline(image=im, mask=ori_gt)
        image = augmented['image']
        mask = augmented['mask']
        return (image, mask, infos)

    def prepare_test_data(self, infos):
        im = cv.imread(infos['filename'])

        # 数据增强
        augmented = self.pipeline(image=im)
        image = augmented['image']
        mask = None
        return (image, mask, infos)

    @staticmethod
    def collate_fn(self, batch):
        images, labels, infos = zip(*batch)
        labels = [x.unsqueeze(0) for x in labels]

        images = torch.stack(images)
        labels = torch.stack(labels)
        batch_infos = {}

        for res in infos:
            del res['return_ori_seg_gt']
            for key, value in res.items():
                if key not in batch_infos:
                    batch_infos[key] = []  # 初始化空列表
                batch_infos[key].append(value)

        # 如果用户没有指定数据集的原始图像尺寸，说明数据集中的原始图像的尺寸不统一
        # 这时，将batch_infos信息字典中的'ori_img_size'设置为list of tuple: [(height_1,width_1),...,(height_n,width_n)]
        # 若在batch_infos中返回原始图像大小的标签，则将多个图像的标签用list of tensor保存 [mask_1,...,mask_n]

        if self.ori_img_size:
            # 如果用户指定了数据集的原始图像尺寸，说明数据集中的原始图像拥有统一的尺寸
            # 这时，将batch_infos信息字典中的'ori_img_size'设置为元组 (height,width)
            # 若在batch_infos中返回原始图像大小的标签，则将多个图像的标签用一个tensor保存
            batch_infos['ori_img_size_hw'] = self.ori_img_size   # tuple
            batch_infos['ori_gt'] = torch.stack(batch_infos['ori_gt'])  # tensor

        return images, labels, batch_infos
