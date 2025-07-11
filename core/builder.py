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
from registry import RegisterManager, RegisterMeta

DATASET = RegisterManager.create_registry('DATASET')
INITIALIZERS = RegisterManager.create_registry('INITIALIZERS')
SAMPLER = RegisterManager.create_registry('SAMPLER')
OPTIMIZER = RegisterManager.create_registry('OPTIMIZER')
LR_SCHEDULER = RegisterManager.create_registry('LR_SCHEDULER')


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


def build_optimizer(cfg, params):
    if not isinstance(cfg, dict):
        raise TypeError('The cfg in build_optimizer function must be a dict.')
    if 'type' not in cfg:
        raise KeyError('The cfg in build_optimizer function must contain the key "type". ')
    cfg_ = cfg.copy()
    optimizer_type = cfg_.pop('type')
    optimizer = OPTIMIZER.get(optimizer_type)
    return optimizer(params=params, **cfg_)
