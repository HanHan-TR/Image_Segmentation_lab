# Copyright (c) OpenMMLab. All rights reserved.
import warnings
from typing import Optional, Union, Sequence
import torch
import numpy as np
import sys
import os
import cv2 as cv
from pathlib import Path, PosixPath
import albumentations as A
from collections import defaultdict
FILE = Path(__file__).resolve()
ROOT = FILE.parents[2]  # root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative
RANK = int(os.getenv('RANK', -1))


from core.fileio import parse_and_backup_config, mkdir_or_exist
from core.fileio.image_io import ImageType
from core.fileio.image_io import imread
from core.initialize import load_checkpoint
from models.common import BaseModule
from models.segmentors import BaseSegmentor
from models.builder import build_segmentor

# from mmseg.registry import MODELS
# from mmseg.structures import SegDataSample
# from mmseg.utils import SampleList, dataset_aliases, get_classes, get_palette
# from mmseg.visualization import SegLocalVisualizer
# from .utils import ImageType, _preprare_data


def init_model(config: Union[str, Path],
               checkpoint: Optional[str] = None,
               device: str = 'cuda:0'):
    """Initialize a segmentor from config file.

    Args:
        config (str, :obj:`Path`, or :obj:`mmengine.Config`): Config file path,
            :obj:`Path`, or the config object.
        checkpoint (str, optional): Checkpoint path. If left as None, the model
            will not load any weights.
        device (str, optional) CPU/CUDA device option. Default 'cuda:0'.
            Use 'cpu' for loading model on CPU.
        cfg_options (dict, optional): Options to override some settings in
            the used config.
    Returns:
        nn.Module: The constructed segmentor.
    """
    # 解析模型配置文件
    if isinstance(config, (str, Path)):
        network_config = parse_and_backup_config(config).pop('model')
    else:
        raise TypeError('config must be a file path, '
                        'but got {}'.format(type(config)))

    # if cfg_options is not None:
    #     config.merge_from_dict(cfg_options)

    # 修改模型配置的权重初始化相关参数
    if network_config.get('type') == 'EncoderDecoder':
        if 'init_cfg' in network_config['backbone']:
            network_config['backbone']['init_cfg'] = None
    elif network_config.get('type') == 'MultimodalEncoderDecoder':
        for k, v in network_config.items():
            if isinstance(v, dict) and 'init_cfg' in v:
                network_config[k].init_cfg = None
    network_config['pretrained'] = None
    # network_config.train_cfg = None
    # init_default_scope(config.get('default_scope', 'mmseg'))

    # model = MODELS.build(config.model)
    # 构建模型
    model = build_segmentor(cfg=network_config)

    # 加载模型权重
    if checkpoint is not None:
        checkpoint = load_checkpoint(model, checkpoint, map_location='cpu')
        metadata = checkpoint.get('metadata', None)
        # save the dataset_meta in the model for convenience
        if metadata:
            classes = metadata['CLASSES']
            palette = metadata['PALETTE']
            model.metadata = {'classes': classes, 'palette': palette}
    model.cfg = network_config  # save the config in the model for convenience
    model.to(device)
    model.eval()
    return model


def inference_model(model: BaseSegmentor,
                    img: Union[Union[str, np.ndarray], Sequence[Union[str, np.ndarray]]],
                    pipeline: str,
                    device='cuda'):  # -> Union[SegDataSample, SampleList]:
    """使用分割器进行模型推理.

    Args:
        model (nn.Module): 已加载的分割器模型.
        imgs (str/ndarray or list[str/ndarray]): 可以是图像文件路径或者已加载的图像数据.

    Returns:
        :obj:`SegDataSample` or list[:obj:`SegDataSample`]:
        如果imgs是列表或元组, 将返回相同长度的列表类型结果, 否则直接返回分割结果.
    """
    # prepare data
    images, data_infos, is_batch = _preprare_data(img, pipeline=pipeline, device=device)

    # forward the model
    with torch.no_grad():
        # results = model.test_step(images)
        # ! forward_test
        results = model(images, img_metas=data_infos, rescale=True, return_loss=False)

    seg_pred = results.argmax(dim=1)

    seg_pred = seg_pred.cpu().numpy()

    return seg_pred


def _preprare_data(imgs: ImageType,
                   pipeline: Union[str, PosixPath],
                   device):
    # a pipeline for each inference
    # ! albumentations.Compose 的 __call__ 方法（即 pipeline(image=image)）
    # ! 要求 image 是一个 ​单张图像​（NumPy 数组，形状为 (H, W, C)），而不是一个图像列表。
    pipeline = A.load(filepath_or_buffer=pipeline, data_format='yaml') if pipeline else None

    data_infos = defaultdict(list)
    is_batch = True  # 表示有多张图像
    if not isinstance(imgs, (list, tuple)):
        imgs = [imgs]
        is_batch = False  # 表示只有一张图像

    images = []
    for img in imgs:
        if isinstance(img, np.ndarray):  # 已加载的图像数据
            data_infos['ori_img_size_hw'].append(img.shape[:2])
            augmented = pipeline(image=img)
            im = augmented['image']
        else:  # 图像路径
            data_infos['img_file_path'].append(img)

            img = cv.imread(filename=img, flags=cv.COLOR_BGR2RGB)
            data_infos['ori_img_size_hw'].append(img.shape[:2])

            augmented = pipeline(image=img)
            im = augmented['image'].to(device)

        images.append(im)

    return images, data_infos, is_batch


def show_result_pyplot(model: BaseSegmentor,
                       img: Union[str, np.ndarray],
                       result,  # : SegDataSample,
                       opacity: float = 0.5,
                       title: str = '',
                       draw_gt: bool = True,
                       draw_pred: bool = True,
                       wait_time: float = 0,
                       show: bool = True,
                       with_labels: Optional[bool] = True,
                       save_dir=None,
                       out_file=None):
    """Visualize the segmentation results on the image.

    Args:
        model (nn.Module): The loaded segmentor.
        img (str or np.ndarray): Image filename or loaded image.
        result (SegDataSample): The prediction SegDataSample result.
        opacity(float): Opacity of painted segmentation map.
            Default 0.5. Must be in (0, 1] range.
        title (str): The title of pyplot figure.
            Default is ''.
        draw_gt (bool): Whether to draw GT SegDataSample. Default to True.
        draw_pred (bool): Whether to draw Prediction SegDataSample.
            Defaults to True.
        wait_time (float): The interval of show (s). 0 is the special value
            that means "forever". Defaults to 0.
        show (bool): Whether to display the drawn image.
            Default to True.
        with_labels(bool, optional): Add semantic labels in visualization
            result, Default to True.
        save_dir (str, optional): Save file dir for all storage backends.
            If it is None, the backend storage will not save any data.
        out_file (str, optional): Path to output file. Default to None.



    Returns:
        np.ndarray: the drawn image which channel is RGB.
    """
    if hasattr(model, 'module'):
        model = model.module
    if isinstance(img, str):
        image = imread(img, channel_order='rgb')
    else:
        image = img
    if save_dir is not None:
        mkdir_or_exist(save_dir)
    # init visualizer
    # visualizer = SegLocalVisualizer(vis_backends=[dict(type='LocalVisBackend')],
    #                                 save_dir=save_dir,
    #                                 alpha=opacity)
    # visualizer.dataset_meta = dict(
    #     classes=model.dataset_meta['classes'],
    #     palette=model.dataset_meta['palette'])
    # visualizer.add_datasample(name=title,
    #                           image=image,
    #                           data_sample=result,
    #                           draw_gt=draw_gt,
    #                           draw_pred=draw_pred,
    #                           wait_time=wait_time,
    #                           out_file=out_file,
    #                           show=show,
    #                           with_labels=with_labels)
    # vis_img = visualizer.get_image()

    # return vis_img
