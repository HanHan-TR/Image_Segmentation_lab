# Copyright (c) OpenMMLab. All rights reserved.
import warnings
from typing import Optional, Union
import torch

import sys
import os
from pathlib import Path

FILE = Path(__file__).resolve()
ROOT = FILE.parents[2]  # root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative
RANK = int(os.getenv('RANK', -1))

# import mmcv
# import numpy as np
# from mmengine import Config
# from mmengine.registry import init_default_scope
# from mmengine.runner import load_checkpoint
# from mmengine.utils import mkdir_or_exist

from core.fileio import file2dict
from core.initialize import load_checkpoint
from models.segmentors import BaseSegmentor
from core.registry import build_segmentor

from mmseg.registry import MODELS
from mmseg.structures import SegDataSample
from mmseg.utils import SampleList, dataset_aliases, get_classes, get_palette
from mmseg.visualization import SegLocalVisualizer
from .utils import ImageType, _preprare_data


def init_model(config: Union[str, Path],
               checkpoint: Optional[str] = None,
               device: str = 'cuda:0',
               cfg_options: Optional[dict] = None):
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
        network_config = file2dict(config).pop('model')
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
        dataset_meta = checkpoint['meta'].get('dataset_meta', None)
        # save the dataset_meta in the model for convenience
        if 'dataset_meta' in checkpoint.get('meta', {}):
            # mmseg 1.x
            model.dataset_meta = dataset_meta
        elif 'CLASSES' in checkpoint.get('meta', {}):
            # < mmseg 1.x
            classes = checkpoint['meta']['CLASSES']
            palette = checkpoint['meta']['PALETTE']
            model.dataset_meta = {'classes': classes, 'palette': palette}
        else:
            warnings.simplefilter('once')
            warnings.warn('dataset_meta or class names are not saved in the '
                          'checkpoint\'s meta data, classes and palette will be'
                          'set according to num_classes ')
            num_classes = model.decode_head.num_classes
            dataset_name = None
            for name in dataset_aliases.keys():
                if len(get_classes(name)) == num_classes:
                    dataset_name = name
                    break
            if dataset_name is None:
                warnings.warn('No suitable dataset found, use Cityscapes by default')
                dataset_name = 'cityscapes'
            model.dataset_meta = {'classes': get_classes(dataset_name),
                                  'palette': get_palette(dataset_name)
                                  }
    model.cfg = network_config  # save the config in the model for convenience
    model.to(device)
    model.eval()
    return model


def inference_model(model: BaseSegmentor,
                    img: ImageType) -> Union[SegDataSample, SampleList]:
    """Inference image(s) with the segmentor.

    Args:
        model (nn.Module): The loaded segmentor.
        imgs (str/ndarray or list[str/ndarray]): Either image files or loaded
            images.

    Returns:
        :obj:`SegDataSample` or list[:obj:`SegDataSample`]:
        If imgs is a list or tuple, the same length list type results
        will be returned, otherwise return the segmentation results directly.
    """
    # prepare data
    data, is_batch = _preprare_data(img, model)

    # forward the model
    with torch.no_grad():
        results = model.test_step(data)

    return results if is_batch else results[0]


def show_result_pyplot(model: BaseSegmentor,
                       img: Union[str, np.ndarray],
                       result: SegDataSample,
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
        image = mmcv.imread(img, channel_order='rgb')
    else:
        image = img
    if save_dir is not None:
        mkdir_or_exist(save_dir)
    # init visualizer
    visualizer = SegLocalVisualizer(
        vis_backends=[dict(type='LocalVisBackend')],
        save_dir=save_dir,
        alpha=opacity)
    visualizer.dataset_meta = dict(
        classes=model.dataset_meta['classes'],
        palette=model.dataset_meta['palette'])
    visualizer.add_datasample(name=title,
                              image=image,
                              data_sample=result,
                              draw_gt=draw_gt,
                              draw_pred=draw_pred,
                              wait_time=wait_time,
                              out_file=out_file,
                              show=show,
                              with_labels=with_labels)
    vis_img = visualizer.get_image()

    return vis_img
