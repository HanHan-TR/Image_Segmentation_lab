from .basic import *
from .common import *
from .utils import *
from .backbones import *
from .decode_heads import *
from .losses import *
from .segmentors import *

from .builder import build_segmentor


__all__ = [
    'build_segmentor'
]
