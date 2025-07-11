import torch
import torch.nn as nn

import sys
import os
from pathlib import Path

FILE = Path(__file__).resolve()
ROOT = FILE.parents[2]  # root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative
RANK = int(os.getenv('RANK', -1))

from models.builder import DROPOUT


def drop_path(x, drop_prob=0., training=False):
    """Drop paths (Stochastic Depth) per sample (when applied in main path of
    residual blocks).

    We follow the implementation
    https://github.com/rwightman/pytorch-image-models/blob/a2727c1bf78ba0d7b5727f5f95e37fb7f8866b1f/timm/models/layers/drop.py  # noqa: E501
    """
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    # handle tensors with different dimensions, not just 4D tensors.
    shape = (x.shape[0], ) + (1, ) * (x.ndim - 1)
    random_tensor = keep_prob + torch.rand(
        shape, dtype=x.dtype, device=x.device)
    output = x.div(keep_prob) * random_tensor.floor()
    return output


@DROPOUT.register()
class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample  (when applied in main path of
    residual blocks).

    We follow the implementation
    https://github.com/rwightman/pytorch-image-models/blob/a2727c1bf78ba0d7b5727f5f95e37fb7f8866b1f/timm/models/layers/drop.py  # noqa: E501

    Args:
        drop_prob (float): Probability of the path to be zeroed. Default: 0.1
    """

    def __init__(self, drop_prob=0.1):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)


@DROPOUT.register()
def Dropout(*args, **kwargs):
    return nn.Dropout(*args, **kwargs)


@DROPOUT.register()
def Dropout1d(*args, **kwargs):
    return nn.Dropout1d(*args, **kwargs)


@DROPOUT.register()
def Dropout2d(*args, **kwargs):
    return nn.Dropout2d(*args, **kwargs)


@DROPOUT.register()
def Dropout3d(*args, **kwargs):
    return nn.Dropout3d(*args, **kwargs)


@DROPOUT.register()
def AlphaDropout(*args, **kwargs):
    return nn.AlphaDropout(*args, **kwargs)


@DROPOUT.register()
def FeatureAlphaDropout(*args, **kwargs):
    return nn.FeatureAlphaDropout(*args, **kwargs)
