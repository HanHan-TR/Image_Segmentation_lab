import inspect
import copy
import re

import torch.nn as nn
from torch.nn.modules.batchnorm import _BatchNorm
from torch.nn.modules.instancenorm import _InstanceNorm

import sys
import os
from pathlib import Path

FILE = Path(__file__).resolve()
ROOT = FILE.parents[1]  # root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative
RANK = int(os.getenv('RANK', -1))

# from core.registry.register import RegisterMeta
from core.registry import (RegisterMeta,
                           ACTIVATION, NORMALIZATION, CONVOLUTION, PADDING, DROPOUT, LOSS,
                           PLUGINS,
                           SAMPLER,
                           SEGMENTOR,
                           OPTIMIZER)


def build_conv_layer(cfg, *args, **kwargs):
    """Build convolution layer.

    Args:
        cfg (None or dict): The conv layer config, which should contain:
            - type (str): Layer type.
            - layer args: Args needed to instantiate an conv layer.
        args (argument list): Arguments passed to the `__init__` method of the corresponding conv layer.
        kwargs (keyword arguments): Keyword arguments passed to the `__init__` method of the corresponding conv layer.

    Returns:
        nn.Module: Created conv layer.
    """
    # Step 1： ------------------------------- load cfg --------------------------
    if cfg is None:
        cfg_ = dict(type='Conv2d')
    else:
        if not isinstance(cfg, dict):
            raise TypeError('The conv layer cfg must be a dict')
        if 'type' not in cfg:
            raise KeyError('The conv layer cfg dict must contain the key "type"')
        cfg_ = cfg.copy()
    layer_type = cfg_.pop('type')
    # if layer_type not in CONV_LAYERS:
    #     raise KeyError(f'Unrecognized layer type {layer_type}')
    # else:
    #     conv_layer = eval(layer_type)
    # Step 2： ------------------------------- create conv layer --------------------------
    conv_layer = CONVOLUTION.get(layer_type)
    layer = conv_layer(*args, **kwargs, **cfg_)

    return layer


def infer_bn_abbr(class_type):
    """Infer abbreviation from the class name.

    When we build a norm layer with `build_norm_layer()`, we want to preserve
    the norm type in variable names, e.g, self.bn1, self.gn. This method will
    infer the abbreviation to map class types to abbreviations.

    Rule 1: If the class has the property "_abbr_", return the property.
    Rule 2: If the parent class is _BatchNorm, GroupNorm, LayerNorm or
    InstanceNorm, the abbreviation of this layer will be "bn", "gn", "ln" and
    "in" respectively.
    Rule 3: If the class name contains "batch", "group", "layer" or "instance",
    the abbreviation of this layer will be "bn", "gn", "ln" and "in"
    respectively.
    Rule 4: Otherwise, the abbreviation falls back to "norm".

    Args:
        class_type (type): The norm layer type.

    Returns:
        str: The inferred abbreviation.
    """
    if not inspect.isclass(class_type):
        raise TypeError(
            f'class_type must be a type, but got {type(class_type)}')
    if hasattr(class_type, '_abbr_'):
        return class_type._abbr_
    if issubclass(class_type, _InstanceNorm):  # IN is a subclass of BN
        return 'in'
    elif issubclass(class_type, _BatchNorm):
        return 'bn'
    elif issubclass(class_type, nn.GroupNorm):
        return 'gn'
    elif issubclass(class_type, nn.LayerNorm):
        return 'ln'
    else:
        class_name = class_type.__name__.lower()
        if 'batch' in class_name:
            return 'bn'
        elif 'group' in class_name:
            return 'gn'
        elif 'layer' in class_name:
            return 'ln'
        elif 'instance' in class_name:
            return 'in'
        else:
            return 'norm_layer'


def build_norm_layer(cfg, num_features, postfix=''):
    """Build normalization layer.

    Args:
        cfg (dict): The norm layer config, which should contain:
            - type (str): Layer type.
            - layer args: Args needed to instantiate a norm layer.
            - requires_grad (bool, optional): Whether stop gradient updates.
        num_features (int): Number of input channels.
        postfix (int | str): The postfix to be appended into norm abbreviation
            to create named layer.

    Returns:
        tuple [str, nn.Module]: The first element is the layer name consisting
        of abbreviation and postfix, e.g., bn1, gn. The second element is the
        created norm layer.
    """

    # Step 1： ------------------------------- load cfg --------------------------
    if not isinstance(cfg, dict):
        raise TypeError('The norm layer cfg must be a dict')
    if 'type' not in cfg:
        raise KeyError('The norm layer cfg dict must contain the key "type"')
    cfg_ = cfg.copy()

    layer_type = cfg_.pop('type')
    # if layer_type not in NORM_LAYERS:
    #     raise KeyError(f'Unrecognized norm type {layer_type}')

    requires_grad = cfg_.pop('requires_grad', True)

    # Step 2： ------------------------------- create norm layer --------------------------
    # norm_layer = eval(layer_type)('')  # instantiate an norm layer
    norm_layer = NORMALIZATION.get(name=layer_type)  # instantiate an norm layer

    cfg_.setdefault('eps', 1e-5)
    if layer_type != 'GN':
        layer = norm_layer(num_features, **cfg_)
        if layer_type == 'SyncBN' and hasattr(layer, '_specify_ddp_gpu_num'):
            layer._specify_ddp_gpu_num(1)
    else:
        assert 'num_groups' in cfg_
        layer = norm_layer(num_channels=num_features, **cfg_)

    for param in layer.parameters():
        param.requires_grad = requires_grad

    # Step 3： ------------------------------- infer the layer name --------------------------
    abbr = infer_bn_abbr(layer.__class__)
    assert isinstance(postfix, (int, str))
    name = abbr + str(postfix)

    return name, layer


def build_activation_layer(cfg):
    """Build activation layer.

    Args:
        cfg (dict): The activation layer config, which should contain:

            - type (str): Layer type.
            - layer args: Args needed to instantiate an activation layer.

    Returns:
        nn.Module: Created activation layer.
    """
    # Step 1： ------------------------------- load cfg --------------------------
    if not isinstance(cfg, dict):
        raise TypeError('The activation layer cfg must be a dict')
    if 'type' not in cfg:
        raise KeyError('The activation layer cfg dict must contain the key "type"')
    cfg_ = copy.deepcopy(cfg)
    activation_type = cfg_.pop('type')

    # Step 2： ------------------------------- create activation layer --------------------------
    activation_layer = ACTIVATION.get(activation_type)
    return activation_layer(**cfg_)


def build_padding_layer(cfg, *args, **kwargs):
    """Build padding layer.

    Args:
        cfg (None or dict): The padding layer config, which should contain:
            - type (str): Layer type.
            - layer args: Args needed to instantiate a padding layer.

    Returns:
        nn.Module: Created padding layer.
    """
    # Step 1： ------------------------------- load cfg --------------------------
    if not isinstance(cfg, dict):
        raise TypeError('The padding layer cfg must be a dict')
    if 'type' not in cfg:
        raise KeyError('The padding layer cfg dict must contain the key "type"')
    cfg_ = cfg.copy()
    padding_type = cfg_.pop('type')
    # if padding_type not in PADDING_LAYERS:
    #     raise KeyError(f'Unrecognized padding type {padding_type}.')
    # else:
    #     padding_layer = eval(padding_type)
    # Step 2： ------------------------------- create padding layer --------------------------
    padding_layer = PADDING.get(padding_type)
    return padding_layer(*args, **kwargs, **cfg_)


def build_dropout(cfg):
    """Build dropout layer.

    Args:
        cfg (None or dict): The dropout layer config, which should contain:
            - type (str): Layer type.
            - layer args: Args needed to instantiate a dropout layer.

    Returns:
        nn.Module: Created dropout layer.
    """
    # Step 1： ------------------------------- load cfg --------------------------
    if not isinstance(cfg, dict):
        raise TypeError('The dropout layer cfg must be a dict')
    if 'type' not in cfg:
        raise KeyError('The dropout layer cfg dict must contain the key "type"')
    cfg_ = cfg.copy()
    dropout_type = cfg_.pop('type')

    # Step 2： ------------------------------- create dropout layer --------------------------
    dropout_layer = DROPOUT.get(dropout_type)
    return dropout_layer(**cfg_)

    # return layer
    # return eval(cfg_.pop('type'))(**cfg_)


def build_loss(cfg):
    """Build loss layer.

    Args:
        cfg (None or dict): The loss config, which should contain:
            - type (str): Loss type.
            - loss args: Args needed to instantiate a loss.

    Returns:
        nn.Module: Created a loss.
    """
    # Step 1： ------------------------------- load cfg --------------------------
    if not isinstance(cfg, dict):
        raise TypeError('The loss cfg must be a dict')
    if 'type' not in cfg:
        raise KeyError('The loss cfg dict must contain the key "type"')
    cfg_ = cfg.copy()
    loss_type = cfg_.pop('type')

    # Step 2： ------------------------------- create loss --------------------------
    dropout_layer = LOSS.get(loss_type)
    return dropout_layer(**cfg_)


def infer_plugin_abbr(class_type):
    """Infer abbreviation from the class name.

    This method will infer the abbreviation to map class types to
    abbreviations.

    Rule 1: If the class has the property "abbr", return the property.
    Rule 2: Otherwise, the abbreviation falls back to snake case of class
    name, e.g. the abbreviation of ``FancyBlock`` will be ``fancy_block``.

    Args:
        class_type (type): The norm layer type.

    Returns:
        str: The inferred abbreviation.
    """

    def camel2snack(word):
        """Convert camel case word into snack case.

        Modified from `inflection lib
        <https://inflection.readthedocs.io/en/latest/#inflection.underscore>`_.

        Example::

            >>> camel2snack("FancyBlock")
            'fancy_block'
        """

        word = re.sub(r'([A-Z]+)([A-Z][a-z])', r'\1_\2', word)
        word = re.sub(r'([a-z\d])([A-Z])', r'\1_\2', word)
        word = word.replace('-', '_')
        return word.lower()

    if not inspect.isclass(class_type):
        raise TypeError(
            f'class_type must be a type, but got {type(class_type)}')
    if hasattr(class_type, '_abbr_'):
        return class_type._abbr_
    else:
        return camel2snack(class_type.__name__)


def build_plugin_layer(cfg, postfix='', **kwargs):
    """Build plugin layer.

    Args:
        cfg (None or dict): cfg should contain:

            - type (str): identify plugin layer type.
            - layer args: args needed to instantiate a plugin layer.
        postfix (int, str): appended into norm abbreviation to
            create named layer. Default: ''.

    Returns:
        tuple[str, nn.Module]: The first one is the concatenation of
        abbreviation and postfix. The second is the created plugin layer.
    """
    if not isinstance(cfg, dict):
        raise TypeError('plugin layer cfg must be a dict')
    if 'type' not in cfg:
        raise KeyError('the plugin layer cfg dict must contain the key "type"')
    cfg_ = cfg.copy()

    layer_type = cfg_.pop('type')

    plugin_layer = PLUGINS.get(layer_type)
    abbr = infer_plugin_abbr(plugin_layer)

    assert isinstance(postfix, (int, str))
    name = abbr + str(postfix)

    layer = plugin_layer(**kwargs, **cfg_)

    return name, layer


def build_from_cfg(cfg, registry, default_args=None):
    """Build a module from config dict.

    Args:
        cfg (dict): Config dict. It should at least contain the key "type".
        registry (:obj:`Registry`): The registry to search the type from.
        default_args (dict, optional): Default initialization arguments.

    Returns:
        object: The constructed object.
    """
    if not isinstance(cfg, dict):
        raise TypeError(f'cfg must be a dict, but got {type(cfg)}')
    if 'type' not in cfg:
        if default_args is None or 'type' not in default_args:
            raise KeyError(
                '`cfg` or `default_args` must contain the key "type", '
                f'but got {cfg}\n{default_args}')
    if not isinstance(registry, RegisterMeta):
        raise TypeError(f'registry must be an RegisterMeta object, but got {type(registry)}')
    if not (isinstance(default_args, dict) or default_args is None):
        raise TypeError(f'default_args must be a dict or None, but got {type(default_args)}')

    args = cfg.copy()

    if default_args is not None:
        for name, value in default_args.items():
            args.setdefault(name, value)

    obj_type = args.pop('type')
    if isinstance(obj_type, str):
        obj_cls = registry.get(obj_type)
        if obj_cls is None:
            raise KeyError(
                f'{obj_type} is not in the {registry.name} registry')
    else:
        raise TypeError(
            f'config type must be a str type, but got {type(obj_type)}')
    try:
        return obj_cls(**args)
    except Exception as e:
        # Normal TypeError does not print class name.
        raise type(e)(f'{obj_cls.__name__}: {e}')


def build_pixel_sampler(cfg, **default_args):
    """Build pixel sampler for segmentation map."""
    return build_from_cfg(cfg, SAMPLER, default_args)


def build_segmentor(cfg):
    """Build Segmentor

    Args:
        cfg (dict): The segmentor config, which should contain:
            - type (str): Segmentor type.
    """
    if not isinstance(cfg, dict):
        raise TypeError('The cfg in build_segmentor function must be a dict.')
    if 'type' not in cfg:
        raise KeyError('The cfg in build_segmentor function must contain the key "type". ')

    cfg_ = cfg.copy()
    segmentor_type = cfg_.pop('type')
    segmentor = SEGMENTOR.get(segmentor_type)
    return segmentor(**cfg_)


def build_optimizer(cfg, params):
    if not isinstance(cfg, dict):
        raise TypeError('The cfg in build_optimizer function must be a dict.')
    if 'type' not in cfg:
        raise KeyError('The cfg in build_optimizer function must contain the key "type". ')
    cfg_ = cfg.copy()
    optimizer_type = cfg_.pop('type')
    optimizer = OPTIMIZER.get(optimizer_type)
    return optimizer(params=params, **cfg_)
