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

from core.registry import PADDING


@PADDING.register()
def zero(*args, **kwargs):
    return nn.ZeroPad2d(*args, **kwargs)


@PADDING.register()
def reflect(*args, **kwargs):
    return nn.ReflectionPad2d(*args, **kwargs)


@PADDING.register()
def replicate(*args, **kwargs):
    return nn.ReplicationPad2d(*args, **kwargs)
