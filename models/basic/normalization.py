import torch.nn as nn
import torch.nn.functional as F

import sys
import os
from pathlib import Path

FILE = Path(__file__).resolve()
ROOT = FILE.parents[2]  # root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative
RANK = int(os.getenv('RANK', -1))

from models.builder import NORMALIZATION


@NORMALIZATION.register()
def BatchNorm1d(*args, **kwargs):
    return nn.BatchNorm1d(*args, **kwargs)


@NORMALIZATION.register()
def BatchNorm2d(*args, **kwargs):
    return nn.BatchNorm2d(*args, **kwargs)


@NORMALIZATION.register()
def BatchNorm(*args, **kwargs):
    return nn.BatchNorm2d(*args, **kwargs)


@NORMALIZATION.register()
def BatchNorm3d(*args, **kwargs):
    return nn.BatchNorm3d(*args, **kwargs)


@NORMALIZATION.register()
def SyncBatchNorm(*args, **kwargs):
    return nn.SyncBatchNorm(*args, **kwargs)


@NORMALIZATION.register()
def GroupNorm(*args, **kwargs):
    return nn.GroupNorm(*args, **kwargs)


@NORMALIZATION.register()
def LayerNorm(*args, **kwargs):
    return nn.LayerNorm(*args, **kwargs)


@NORMALIZATION.register()
def InstanceNorm2d(*args, **kwargs):
    return nn.InstanceNorm2d(*args, **kwargs)


@NORMALIZATION.register()
def InstanceNorm(*args, **kwargs):
    return nn.InstanceNorm2d(*args, **kwargs)


@NORMALIZATION.register()
def InstanceNorm1d(*args, **kwargs):
    return nn.InstanceNorm1d(*args, **kwargs)


@NORMALIZATION.register()
def InstanceNorm3d(*args, **kwargs):
    return nn.InstanceNorm3d(*args, **kwargs)


@NORMALIZATION.register()
class LayerNorm2d(nn.LayerNorm):
    """LayerNorm on channels for 2d images.

    Args:
        num_channels (int): The number of channels of the input tensor.
        eps (float): a value added to the denominator for numerical stability.
            Defaults to 1e-5.
        elementwise_affine (bool): a boolean value that when set to ``True``,
            this module has learnable per-element affine parameters initialized
            to ones (for weights) and zeros (for biases). Defaults to True.
    """

    def __init__(self, num_channels: int, **kwargs) -> None:
        super().__init__(num_channels, **kwargs)
        self.num_channels = self.normalized_shape[0]

    def forward(self, x, data_format='channel_first'):
        assert x.dim() == 4, 'LayerNorm2d only supports inputs with shape ' \
            f'(N, C, H, W), but got tensor with shape {x.shape}'
        if data_format == 'channel_last':
            x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias,
                             self.eps)
        elif data_format == 'channel_first':
            x = x.permute(0, 2, 3, 1)
            x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias,
                             self.eps)
            # If the output is discontiguous, it may cause some unexpected
            # problem in the downstream tasks
            x = x.permute(0, 3, 1, 2).contiguous()
        return x
