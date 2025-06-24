import sys
import os
from pathlib import Path

FILE = Path(__file__).resolve()
ROOT = FILE.parents[2]  # root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative
RANK = int(os.getenv('RANK', -1))

from .register import RegisterManager, RegisterMeta

# Predefined common registers
ACTIVATION = RegisterManager.create_registry('ACTIVATION')
CONVOLUTION = RegisterManager.create_registry('CONVOLUTION')
DROPOUT = RegisterManager.create_registry('DROPOUT')
NORMALIZATION = RegisterManager.create_registry('NORMALIZATION')
PADDING = RegisterManager.create_registry('PADDING')
PLUGINS = RegisterManager.create_registry('PLUGINS')

BACKBONE = RegisterManager.create_registry('BACKBONE')
NECK = RegisterManager.create_registry('NECK')
DECODEHEAD = RegisterManager.create_registry('DECODEHEAD')
LOSS = RegisterManager.create_registry('LOSS')

INITIALIZERS = RegisterManager.create_registry('INITIALIZERS')

SEGMENTOR = RegisterManager.create_registry('SEGMENTOR')

DATASET = RegisterManager.create_registry('DATASET')
# tools
SAMPLER = RegisterManager.create_registry('SAMPLER')
DATAPROCESSOR = RegisterManager.create_registry('DATAPROCESSOR')

OPTIMIZER = RegisterManager.create_registry('OPTIMIZER')
LR_SCHEDULER = RegisterManager.create_registry('LR_SCHEDULER')


from .builder import (
    build_activation_layer,
    build_conv_layer,
    build_dropout,
    build_loss,
    build_norm_layer,
    build_padding_layer,
    build_pixel_sampler,
    build_plugin_layer,
    build_segmentor,
    build_from_cfg,
    build_optimizer
)


from dataset import *
from core.initialize import *
from models.basic import *
from models.backbones import *
from models.decode_heads import *
from models.losses import *
from models.segmentors.encoder_decoder import *
from core.optimizers import *


__all__ = [
    'build_activation_layer',
    'build_conv_layer',
    'build_dropout',
    'build_loss',
    'build_norm_layer',
    'build_padding_layer',
    'build_pixel_sampler',
    'build_plugin_layer',
    'build_segmentor',
    'build_from_cfg',
    'build_optimizer'
]
