import re
import torch
import os.path as osp
import numpy as np
from typing import Any, Callable
from collections import OrderedDict
from core.mixed_precision import get_dist_info


def load_state_dict(module, state_dict, strict=False, logger=None):
    """将状态字典加载到模块中。

    该方法修改自 :meth:torch.nn.Module.load_state_dict。
    默认将 strict 参数设为 False，且即使严格模式为 False 也会显示参数不匹配的提示信息。

    Args:
        module (Module): 接收权重的模型.
        state_dict (OrderedDict): 权重字典
        strict (bool): 是否严格检查state_dict的键与接收权重文件的模型的
            :meth:`~torch.nn.Module.state_dict` 方法返回的键完全相同。默认值: ``False``，即默认关闭严格模式
        logger (:obj:`logging.Logger`, optional): 用于记录错误信息的日志记录器。若未指定，将使用 print 函数输出

    Note:
        - unexpected_keys (list of str) 用于记录非预期的参数键，指存在于state_dict中但未被模型使用的参数键，即模型当前结构不需要的多余参数
        - all_missing_keys (list of str) 用于记录所有缺失的参数键，指模型需要的但state_dict中缺少的参数键，即当前模型结构中未被初始化的必需参数
        需要注意的是， all_missing_keys和unexpected_keys只会反映出权重的键（名称）不同的情况，而不能反映权重向量维度不匹配的情况.
        有关向量维度不匹配的信息将由PyTorch底层的加载方法自动填充到err_msg中打印输出到终端。
        - err_msg (list of str) 错误信息缓存
    """
    unexpected_keys = []
    all_missing_keys = []
    err_msg = []

    metadata = getattr(state_dict, '_metadata', None)  # 获取模型元数据
    state_dict = state_dict.copy()  # 创建副本避免污染原始数据
    if metadata is not None:
        state_dict._metadata = metadata  # 保持元数据完整

    def load(module, prefix=''):
        local_metadata = {} if metadata is None else metadata.get(prefix[:-1], {})
        # 调用PyTorch底层加载方法
        module._load_from_state_dict(state_dict,
                                     prefix,
                                     local_metadata,
                                     True,
                                     all_missing_keys,
                                     unexpected_keys,
                                     err_msg)

        for name, child in module._modules.items():  # 递归处理子模块
            if child is not None:
                load(child, prefix + name + '.')

    load(module)
    load = None  # 打破 load->load 的引用循环

    missing_keys = [
        # ignore "num_batches_tracked" of BN layers
        key for key in all_missing_keys if 'num_batches_tracked' not in key
    ]

    if unexpected_keys:
        err_msg.append('unexpected key in source '
                       f'state_dict: {", ".join(unexpected_keys)}\n')
    if missing_keys:
        err_msg.append(f'missing keys in source state_dict: {", ".join(missing_keys)}\n')

    rank, _ = get_dist_info()
    if len(err_msg) > 0 and rank == 0:
        err_msg.insert(0, 'The model and loaded state dict do not match exactly\n')
        err_msg = '\n'.join(err_msg)
        if strict:
            raise RuntimeError(err_msg)
        elif logger is not None:
            logger.warning(err_msg)
        else:
            print(err_msg)


def load_from_local(filename, map_location):
    """通过本地文件路径加载检查点。

    Args:
        filename (str): 本地checkpoint文件路径
        map_location (str, optional): 与 :func:`torch.load` 功能相同。

    Returns:
        checkpoint (dict | OrderedDict): 加载的checkpoint数据.
    """
    filename = osp.expanduser(filename)
    if not osp.isfile(filename):
        raise FileNotFoundError(f'{filename} can not be found.')
    checkpoint = torch.load(filename, map_location=map_location)
    return checkpoint


def load_checkpoint(model,
                    filename,
                    map_location=None,
                    strict=False,
                    logger=None,
                    revise_keys=[(r'^module\.', '')]):
    """从本地checkpoint文件加载预训练权重到模型中

    Args:
        model (Module): 待加载预训练权重的模型或模块
        filename (str): 本地checkpoint文件路径
        map_location (str): 与 :func:`torch.load` 功能相同
        strict (bool): 是否允许模型与checkpoint权重的参数不一致
        logger (:mod:`logging.Logger` 或 None): 用于记录错误信息的日志器
        revise_keys (list): 用于修改检查点中state_dict的自定义关键词列表。每个元素为 (pattern, replacement)形式的正则表达式操作对。
        默认行为是通过[(r'^module\\.', '')]去除'module.'前缀。

    Returns:
        dict 或 OrderedDict: 加载的检查点内容
    """
    checkpoint = load_from_local(filename, map_location)
    # OrderedDict is a subclass of dict
    if not isinstance(checkpoint, dict):
        raise RuntimeError(f'No state_dict found in checkpoint file {filename}')
    # get state_dict from checkpoint
    if 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
    else:
        state_dict = checkpoint

    # strip prefix of state_dict
    metadata = getattr(state_dict, '_metadata', OrderedDict())
    for p, r in revise_keys:
        state_dict = OrderedDict(
            {re.sub(p, r, k): v
             for k, v in state_dict.items()})
    # Keep metadata in state_dict
    state_dict._metadata = metadata

    # load state_dict
    load_state_dict(model, state_dict, strict, logger)

    return checkpoint


def load_checkpoint_with_prefix(prefix, filename, map_location=None):
    """Load partial pretrained model with specific prefix.

    Args:
        prefix (str): The prefix of sub-module. For example, if we would like to only load the
            backbone of a detector model, we can set ``prefix='backbone.'``
        filename (str): Accept local filepath
        map_location (str | None): Same as :func:`torch.load`. Default: None.

    Returns:
        dict or OrderedDict: The loaded checkpoint.
    """

    checkpoint = load_from_local(filename, map_location=map_location)

    if 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
    else:
        state_dict = checkpoint
    if not prefix.endswith('.'):
        prefix += '.'
    prefix_len = len(prefix)

    state_dict = {k[prefix_len:]: v
                  for k, v in state_dict.items() if k.startswith(prefix)
                  }

    assert state_dict, f'{prefix} is not in the pretrained model'
    return state_dict


def apply_to(data: Any, expr: Callable, apply_func: Callable):
    """对字典、列表或元组中与表达式匹配的每个元素应用函数。

    例如，如果你想将字典列表中的每个元素从`np.ndarray`转换为`Tensor`，可以使用以下代码:

    Examples:
        >>> from mmengine.utils import apply_to
        >>> import numpy as np
        >>> import torch
        >>> data = dict(array=[np.array(1)]) # {'array': [array(1)]}
        >>> result = apply_to(data, lambda x: isinstance(x, np.ndarray), lambda x: torch.from_numpy(x))
        >>> print(result) # {'array': [tensor(1)]}

    Args:
        data (Any): 待应用的数据.
        expr (Callable): 用于判断哪些数据应被该函数处理的表达式。它应返回一个布尔值.
        apply_func (Callable): 应用于数据的函数.

    Returns:
        Any: 应用后的数据.
    """  # noqa: E501
    if isinstance(data, dict):
        # Keep the original dict type
        res = type(data)()
        for key, value in data.items():
            res[key] = apply_to(value, expr, apply_func)
        return res
    elif isinstance(data, tuple) and hasattr(data, '_fields'):
        # namedtuple
        return type(data)(*(apply_to(sample, expr, apply_func) for sample in data))  # type: ignore  # noqa: E501  # yapf:disable
    elif isinstance(data, (tuple, list)):
        return type(data)(apply_to(sample, expr, apply_func) for sample in data)  # type: ignore  # noqa: E501  # yapf:disable
    elif expr(data):
        return apply_func(data)
    else:
        return data


def weights_to_cpu(state_dict):
    """将模型的state_dict复制到CPU。

    Args:
        state_dict (OrderedDict): 模型在GPU上的权重。

    Returns:
        OrderedDict: CPU上的模型权重
    """
    metadata = getattr(state_dict, '_metadata', OrderedDict())
    state_dict = apply_to(data=state_dict,
                          expr=lambda x: hasattr(x, 'cpu'),
                          apply_func=lambda x: x.cpu())
    state_dict._metadata = metadata
    return state_dict
