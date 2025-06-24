# Copyright (c) OpenMMLab. All rights reserved.
import math

from torch import nn
from torch.nn import functional as F

import sys
import os
from pathlib import Path

FILE = Path(__file__).resolve()
ROOT = FILE.parents[2]  # root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative
RANK = int(os.getenv('RANK', -1))

from core.registry import CONVOLUTION


@CONVOLUTION.register()
def Conv1d(*args, **kwargs):
    return nn.Conv1d(*args, **kwargs)


@CONVOLUTION.register()
def Conv2d(*args, **kwargs):
    return nn.Conv2d(*args, **kwargs)


@CONVOLUTION.register()
def Conv3d(*args, **kwargs):
    return nn.Conv3d(*args, **kwargs)


@CONVOLUTION.register()
def Conv(*args, **kwargs):
    return nn.Conv2d(*args, **kwargs)
