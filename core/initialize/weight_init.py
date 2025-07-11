# Copyright (c) OpenMMLab. All rights reserved.
import copy
import math
import warnings

import numpy as np
import torch
import torch.nn as nn
from torch import Tensor

from core.initialize.init_functions import (_get_bases_name, constant_init, caffe2_xavier_init,
                                            normal_init, xavier_init, kaiming_init, uniform_init, trunc_normal_init,
                                            update_init_info, bias_init_with_prob)
from core.builder import INITIALIZERS, build_from_cfg


class BaseInit(object):

    def __init__(self, *, bias=0, bias_prob=None, layer=None):
        self.wholemodule = False
        if not isinstance(bias, (int, float)):
            raise TypeError(f'bias must be a number, but got a {type(bias)}')

        if bias_prob is not None:
            if not isinstance(bias_prob, float):
                raise TypeError(f'bias_prob type must be float, \
                    but got {type(bias_prob)}')

        if layer is not None:
            if not isinstance(layer, (str, list)):
                raise TypeError(f'layer must be a str or a list of str, \
                    but got a {type(layer)}')
        else:
            layer = []

        if bias_prob is not None:
            self.bias = bias_init_with_prob(bias_prob)
        else:
            self.bias = bias
        self.layer = [layer] if isinstance(layer, str) else layer

    def _get_init_info(self):
        info = f'{self.__class__.__name__}, bias={self.bias}'
        return info


@INITIALIZERS.register()
class ConstantInit(BaseInit):
    """Initialize module parameters with constant values.

    Args:
        val (int | float): the value to fill the weights in the module with
        bias (int | float): the value to fill the bias. Defaults to 0.
        bias_prob (float, optional): the probability for bias initialization.
            Defaults to None.
        layer (str | list[str], optional): the layer will be initialized.
            Defaults to None.
    """

    def __init__(self, val, **kwargs):
        super().__init__(**kwargs)
        self.val = val

    def __call__(self, module):

        def init(m):
            if self.wholemodule:
                constant_init(m, self.val, self.bias)
            else:
                layername = m.__class__.__name__
                basesname = _get_bases_name(m)
                if len(set(self.layer) & set([layername] + basesname)):
                    constant_init(m, self.val, self.bias)

        module.apply(init)
        if hasattr(module, '_params_init_info'):
            update_init_info(module, init_info=self._get_init_info())

    def _get_init_info(self):
        info = f'{self.__class__.__name__}: val={self.val}, bias={self.bias}'
        return info


@INITIALIZERS.register()
class XavierInit(BaseInit):
    r"""Initialize module parameters with values according to the method
    described in `Understanding the difficulty of training deep feedforward
    neural networks - Glorot, X. & Bengio, Y. (2010).
    <http://proceedings.mlr.press/v9/glorot10a/glorot10a.pdf>`_

    Args:
        gain (int | float): an optional scaling factor. Defaults to 1.
        bias (int | float): the value to fill the bias. Defaults to 0.
        bias_prob (float, optional): the probability for bias initialization.
            Defaults to None.
        distribution (str): distribution either be ``'normal'``
            or ``'uniform'``. Defaults to ``'normal'``.
        layer (str | list[str], optional): the layer will be initialized.
            Defaults to None.
    """

    def __init__(self, gain=1, distribution='normal', **kwargs):
        super().__init__(**kwargs)
        self.gain = gain
        self.distribution = distribution

    def __call__(self, module):

        def init(m):
            if self.wholemodule:
                xavier_init(m, self.gain, self.bias, self.distribution)
            else:
                layername = m.__class__.__name__
                basesname = _get_bases_name(m)
                if len(set(self.layer) & set([layername] + basesname)):
                    xavier_init(m, self.gain, self.bias, self.distribution)

        module.apply(init)
        if hasattr(module, '_params_init_info'):
            update_init_info(module, init_info=self._get_init_info())

    def _get_init_info(self):
        info = f'{self.__class__.__name__}: gain={self.gain}, ' \
               f'distribution={self.distribution}, bias={self.bias}'
        return info


@INITIALIZERS.register()
class NormalInit(BaseInit):
    r"""Initialize module parameters with the values drawn from the normal
    distribution :math:`\mathcal{N}(\text{mean}, \text{std}^2)`.

    Args:
        mean (int | float):the mean of the normal distribution. Defaults to 0.
        std (int | float): the standard deviation of the normal distribution.
            Defaults to 1.
        bias (int | float): the value to fill the bias. Defaults to 0.
        bias_prob (float, optional): the probability for bias initialization.
            Defaults to None.
        layer (str | list[str], optional): the layer will be initialized.
            Defaults to None.

    """

    def __init__(self, mean=0, std=1, **kwargs):
        super().__init__(**kwargs)
        self.mean = mean
        self.std = std

    def __call__(self, module):

        def init(m):
            if self.wholemodule:
                normal_init(m, self.mean, self.std, self.bias)
            else:
                layername = m.__class__.__name__
                basesname = _get_bases_name(m)
                if len(set(self.layer) & set([layername] + basesname)):
                    normal_init(m, self.mean, self.std, self.bias)

        module.apply(init)
        if hasattr(module, '_params_init_info'):
            update_init_info(module, init_info=self._get_init_info())

    def _get_init_info(self):
        info = f'{self.__class__.__name__}: mean={self.mean},' \
               f' std={self.std}, bias={self.bias}'
        return info


@INITIALIZERS.register()
class TruncNormalInit(BaseInit):
    r"""Initialize module parameters with the values drawn from the normal
    distribution :math:`\mathcal{N}(\text{mean}, \text{std}^2)` with values
    outside :math:`[a, b]`.

    Args:
        mean (float): the mean of the normal distribution. Defaults to 0.
        std (float):  the standard deviation of the normal distribution.
            Defaults to 1.
        a (float): The minimum cutoff value.
        b ( float): The maximum cutoff value.
        bias (float): the value to fill the bias. Defaults to 0.
        bias_prob (float, optional): the probability for bias initialization.
            Defaults to None.
        layer (str | list[str], optional): the layer will be initialized.
            Defaults to None.

    """

    def __init__(self,
                 mean: float = 0,
                 std: float = 1,
                 a: float = -2,
                 b: float = 2,
                 **kwargs) -> None:
        super().__init__(**kwargs)
        self.mean = mean
        self.std = std
        self.a = a
        self.b = b

    def __call__(self, module: nn.Module) -> None:

        def init(m):
            if self.wholemodule:
                trunc_normal_init(m, self.mean, self.std, self.a, self.b,
                                  self.bias)
            else:
                layername = m.__class__.__name__
                basesname = _get_bases_name(m)
                if len(set(self.layer) & set([layername] + basesname)):
                    trunc_normal_init(m, self.mean, self.std, self.a, self.b,
                                      self.bias)

        module.apply(init)
        if hasattr(module, '_params_init_info'):
            update_init_info(module, init_info=self._get_init_info())

    def _get_init_info(self):
        info = f'{self.__class__.__name__}: a={self.a}, b={self.b},' \
               f' mean={self.mean}, std={self.std}, bias={self.bias}'
        return info


@INITIALIZERS.register()
class UniformInit(BaseInit):
    r"""Initialize module parameters with values drawn from the uniform
    distribution :math:`\mathcal{U}(a, b)`.

    Args:
        a (int | float): the lower bound of the uniform distribution.
            Defaults to 0.
        b (int | float): the upper bound of the uniform distribution.
            Defaults to 1.
        bias (int | float): the value to fill the bias. Defaults to 0.
        bias_prob (float, optional): the probability for bias initialization.
            Defaults to None.
        layer (str | list[str], optional): the layer will be initialized.
            Defaults to None.
    """

    def __init__(self, a=0, b=1, **kwargs):
        super().__init__(**kwargs)
        self.a = a
        self.b = b

    def __call__(self, module):

        def init(m):
            if self.wholemodule:
                uniform_init(m, self.a, self.b, self.bias)
            else:
                layername = m.__class__.__name__
                basesname = _get_bases_name(m)
                if len(set(self.layer) & set([layername] + basesname)):
                    uniform_init(m, self.a, self.b, self.bias)

        module.apply(init)
        if hasattr(module, '_params_init_info'):
            update_init_info(module, init_info=self._get_init_info())

    def _get_init_info(self):
        info = f'{self.__class__.__name__}: a={self.a},' \
               f' b={self.b}, bias={self.bias}'
        return info


@INITIALIZERS.register()
class KaimingInit(BaseInit):
    r"""Initialize module parameters with the values according to the method
    described in `Delving deep into rectifiers: Surpassing human-level
    performance on ImageNet classification - He, K. et al. (2015).
    <https://www.cv-foundation.org/openaccess/content_iccv_2015/
    papers/He_Delving_Deep_into_ICCV_2015_paper.pdf>`_

    Args:
        a (int | float): the negative slope of the rectifier used after this
            layer (only used with ``'leaky_relu'``). Defaults to 0.
        mode (str):  either ``'fan_in'`` or ``'fan_out'``. Choosing
            ``'fan_in'`` preserves the magnitude of the variance of the weights
            in the forward pass. Choosing ``'fan_out'`` preserves the
            magnitudes in the backwards pass. Defaults to ``'fan_out'``.
        nonlinearity (str): the non-linear function (`nn.functional` name),
            recommended to use only with ``'relu'`` or ``'leaky_relu'`` .
            Defaults to 'relu'.
        bias (int | float): the value to fill the bias. Defaults to 0.
        bias_prob (float, optional): the probability for bias initialization.
            Defaults to None.
        distribution (str): distribution either be ``'normal'`` or
            ``'uniform'``. Defaults to ``'normal'``.
        layer (str | list[str], optional): the layer will be initialized.
            Defaults to None.
    """

    def __init__(self,
                 a=0,
                 mode='fan_out',
                 nonlinearity='relu',
                 distribution='normal',
                 **kwargs):
        super().__init__(**kwargs)
        self.a = a
        self.mode = mode
        self.nonlinearity = nonlinearity
        self.distribution = distribution

    def __call__(self, module):

        def init(m):
            if self.wholemodule:
                kaiming_init(m, self.a, self.mode, self.nonlinearity,
                             self.bias, self.distribution)
            else:
                layername = m.__class__.__name__
                basesname = _get_bases_name(m)
                if len(set(self.layer) & set([layername] + basesname)):
                    kaiming_init(m, self.a, self.mode, self.nonlinearity,
                                 self.bias, self.distribution)

        module.apply(init)
        if hasattr(module, '_params_init_info'):
            update_init_info(module, init_info=self._get_init_info())

    def _get_init_info(self):
        info = f'{self.__class__.__name__}: a={self.a}, mode={self.mode}, ' \
               f'nonlinearity={self.nonlinearity}, ' \
               f'distribution ={self.distribution}, bias={self.bias}'
        return info


@INITIALIZERS.register()
class Caffe2XavierInit(KaimingInit):
    # `XavierFill` in Caffe2 corresponds to `kaiming_uniform_` in PyTorch
    # Acknowledgment to FAIR's internal code
    def __init__(self, **kwargs):
        super().__init__(
            a=1,
            mode='fan_in',
            nonlinearity='leaky_relu',
            distribution='uniform',
            **kwargs)

    def __call__(self, module):
        super().__call__(module)


@INITIALIZERS.register()
class PretrainedInit(object):
    """通过加载预训练模型权重来初始化模型

    Args:
        checkpoint (str): 需要加载的预训练模型的 checkpoint 文件
        prefix (str, optional): 预训练模型中子模块的前缀。用于仅加载预训练模型的一部分进行初始化。
        例如，若只需加载检测器模型的主干网络，可设置``prefix='backbone.'``。默认为None
        map_location (str): 将张量映射到指定设备
    """

    def __init__(self, checkpoint, prefix=None, map_location=None):
        self.checkpoint = checkpoint
        self.prefix = prefix
        self.map_location = map_location

    def __call__(self, module):
        from .checkpoint import (load_checkpoint_with_prefix, load_checkpoint,
                                 load_state_dict)
        if self.prefix is None:
            print(f'load model from: {self.checkpoint}')
            load_checkpoint(module,
                            self.checkpoint,
                            map_location=self.map_location,
                            strict=False)
        else:
            print(f'load {self.prefix} in model from: {self.checkpoint}')
            state_dict = load_checkpoint_with_prefix(self.prefix, self.checkpoint, map_location=self.map_location)
            load_state_dict(module, state_dict, strict=False)

        if hasattr(module, '_params_init_info'):
            update_init_info(module, init_info=self._get_init_info())

    def _get_init_info(self):
        info = f'{self.__class__.__name__}: load from {self.checkpoint}'
        return info


def _initialize(module, cfg, wholemodule=False):
    """wholemodule标志用于覆盖模式.

    覆盖模式下,将丢弃初始化配置cfg字典中的layer键, 初始化器会为覆盖模式中指定名称的模块提供以初始值。
    """
    func = build_from_cfg(cfg, INITIALIZERS)
    func.wholemodule = wholemodule
    func(module)


def _initialize_override(module, override, cfg):
    if not isinstance(override, (dict, list)):
        raise TypeError(f'override must be a dict or a list of dict, \
                but got {type(override)}')

    override = [override] if isinstance(override, dict) else override

    for override_ in override:

        cp_override = copy.deepcopy(override_)
        name = cp_override.pop('name', None)
        if name is None:
            raise ValueError(f'`override` must contain the key "name", but got {cp_override}')
        # if override only has name key, it means use args in init_cfg

        if not cp_override:
            cp_override.update(cfg)
        # if override has name key and other args except type key, it will
        # raise error
        elif 'type' not in cp_override.keys():
            raise ValueError(
                f'`override` need "type" key, but got {cp_override}')

        if hasattr(module, name):
            _initialize(getattr(module, name), cp_override, wholemodule=True)
        else:
            raise RuntimeError(f'module did not have attribute {name}, '
                               f'but init_cfg is {cp_override}.')


def initialize(module, init_cfg):
    """初始化一个模块。

    Args:
        module (``torch.nn.Module``): 待初始化的模块.
        init_cfg (dict | list[dict]): 初始化配置字典，用于定义初始化方式。目前已实现6种初始化器, 包括:
        ``Constant``, ``Xavier``, ``Normal``, ``Uniform``, ``Kaiming``, 以及 ``Pretrained``.

    Example:
        >>> module = nn.Linear(2, 3, bias=True)
        >>> init_cfg = dict(type='Constant', layer='Linear', val =1 , bias =2)
        >>> initialize(module, init_cfg)

        >>> # 定义键值 `'layer'` 以使用不同配置初始化不同类型的网络层
        >>> module = nn.Sequential(nn.Conv1d(3, 1, 3), nn.Linear(1,2))
        >>> init_cfg = [dict(type='Constant', layer='Conv1d', val=1),
                        dict(type='Constant', layer='Linear', val=2)]
        >>> initialize(module, init_cfg)

        >>> # 定义键值`override`以初始化模块中的特定部分
        >>> class FooNet(nn.Module):
        >>>     def __init__(self):
        >>>         super().__init__()
        >>>         self.feat = nn.Conv2d(3, 16, 3)
        >>>         self.reg = nn.Conv2d(16, 10, 3)
        >>>         self.cls = nn.Conv2d(16, 5, 3)
        >>> model = FooNet()
        >>> init_cfg = dict(type='Constant', val=1, bias=2, layer='Conv2d',
        >>>     override=dict(type='Constant', name='reg', val=3, bias=4))
        >>> initialize(model, init_cfg)

        >>> # 使用预训练模型初始化权重。
        >>> model = ResNet(depth=50)
        >>> init_cfg = dict(type='Pretrained',
                checkpoint='torchvision://resnet50')
        >>> initialize(model, init_cfg)

        >>> # 使用"prefix"前缀从预训练模型的特定部分初始化子模块权重
        >>> file = 'retinanet_r50_fpn_1x_coco_20200130-c2398f9e.pth'
        >>> init_cfg = dict(type='Pretrained', checkpoint=file, prefix='backbone.')
    """
    if not isinstance(init_cfg, (dict, list)):
        raise TypeError(f'init_cfg must be a dict or a list of dict, but got {type(init_cfg)}')

    if isinstance(init_cfg, dict):
        init_cfg = [init_cfg]

    for cfg in init_cfg:
        # 应当对原始配置进行深拷贝，因为cfg可能被其他模块使用（例如多个Bottleneck共享的init_cfg），弹出操作会修改
        # 预期配置值，从而影响其他模块的初始化行为。
        cp_cfg = copy.deepcopy(cfg)
        override = cp_cfg.pop('override', None)
        _initialize(module, cp_cfg)

        if override is not None:  # !! 以覆盖模式初始化模块，即为指定名称的模块进行相应初始化
            cp_cfg.pop('layer', None)
            _initialize_override(module, override, cp_cfg)
        else:
            # All attributes in module have same initialization.
            pass


def _no_grad_trunc_normal_(tensor: Tensor, mean: float, std: float, a: float,
                           b: float) -> Tensor:
    # Method based on
    # https://people.sc.fsu.edu/~jburkardt/presentations/truncated_normal.pdf
    # Modified from
    # https://github.com/pytorch/pytorch/blob/master/torch/nn/init.py
    def norm_cdf(x):
        # Computes standard normal cumulative distribution function
        return (1. + math.erf(x / math.sqrt(2.))) / 2.

    if (mean < a - 2 * std) or (mean > b + 2 * std):
        warnings.warn(
            'mean is more than 2 std from [a, b] in nn.init.trunc_normal_. '
            'The distribution of values may be incorrect.',
            stacklevel=2)

    with torch.no_grad():
        # Values are generated by using a truncated uniform distribution and
        # then using the inverse CDF for the normal distribution.
        # Get upper and lower cdf values
        lower = norm_cdf((a - mean) / std)
        upper = norm_cdf((b - mean) / std)

        # Uniformly fill tensor with values from [lower, upper], then translate
        # to [2lower-1, 2upper-1].
        tensor.uniform_(2 * lower - 1, 2 * upper - 1)

        # Use inverse cdf transform for normal distribution to get truncated
        # standard normal
        tensor.erfinv_()

        # Transform to proper mean, std
        tensor.mul_(std * math.sqrt(2.))
        tensor.add_(mean)

        # Clamp to ensure it's in the proper range
        tensor.clamp_(min=a, max=b)
        return tensor


def trunc_normal_(tensor: Tensor,
                  mean: float = 0.,
                  std: float = 1.,
                  a: float = -2.,
                  b: float = 2.) -> Tensor:
    r"""Fills the input Tensor with values drawn from a truncated
    normal distribution. The values are effectively drawn from the
    normal distribution :math:`\mathcal{N}(\text{mean}, \text{std}^2)`
    with values outside :math:`[a, b]` redrawn until they are within
    the bounds. The method used for generating the random values works
    best when :math:`a \leq \text{mean} \leq b`.

    Modified from
    https://github.com/pytorch/pytorch/blob/master/torch/nn/init.py

    Args:
        tensor (``torch.Tensor``): an n-dimensional `torch.Tensor`.
        mean (float): the mean of the normal distribution.
        std (float): the standard deviation of the normal distribution.
        a (float): the minimum cutoff value.
        b (float): the maximum cutoff value.
    """
    return _no_grad_trunc_normal_(tensor, mean, std, a, b)
