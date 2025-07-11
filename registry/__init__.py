
from .register import RegisterManager, RegisterMeta


__all__ = [
    'RegisterManager', 'RegisterMeta'
]


# Predefined common registers
# # ---------------------- models -----------------------------------
# ACTIVATION = RegisterManager.create_registry('ACTIVATION')
# CONVOLUTION = RegisterManager.create_registry('CONVOLUTION')
# DROPOUT = RegisterManager.create_registry('DROPOUT')
# NORMALIZATION = RegisterManager.create_registry('NORMALIZATION')
# PADDING = RegisterManager.create_registry('PADDING')
# PLUGINS = RegisterManager.create_registry('PLUGINS')

# BACKBONE = RegisterManager.create_registry('BACKBONE')
# NECK = RegisterManager.create_registry('NECK')
# DECODEHEAD = RegisterManager.create_registry('DECODEHEAD')
# SEGMENTOR = RegisterManager.create_registry('SEGMENTOR')
# LOSS = RegisterManager.create_registry('LOSS')
# ------------------- dataset -------------------------------------
# DATASET = RegisterManager.create_registry('DATASET')

# ----------------- core -----------------------------------------
# INITIALIZERS = RegisterManager.create_registry('INITIALIZERS')
# SAMPLER = RegisterManager.create_registry('SAMPLER')
# OPTIMIZER = RegisterManager.create_registry('OPTIMIZER')
# LR_SCHEDULER = RegisterManager.create_registry('LR_SCHEDULER')


# from .builder import (
#     build_activation_layer,
#     build_conv_layer,
#     build_dropout,
#     build_loss,
#     build_norm_layer,
#     build_padding_layer,
#     build_pixel_sampler,
#     build_plugin_layer,
#     build_segmentor,
#     build_from_cfg,
#     build_optimizer
# )


# from dataset import *
# from core.initialize import *
# from models.basic import *
# from models.common import *
# from models.utils import *
# from models.backbones import *
# from models.decode_heads import *
# from models.losses import *
# from models.segmentors.encoder_decoder import *
# from core.optimizers import *


# __all__ = [
#     'build_activation_layer',
#     'build_conv_layer',
#     'build_dropout',
#     'build_loss',
#     'build_norm_layer',
#     'build_padding_layer',
#     'build_pixel_sampler',
#     'build_plugin_layer',
#     'build_segmentor',
#     'build_from_cfg',
#     'build_optimizer'
# ]
