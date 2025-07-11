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

from models.builder import ACTIVATION


@ACTIVATION.register()
def ReLU(*args, **kwargs):
    return nn.ReLU(*args, **kwargs)


@ACTIVATION.register()
def ReLU6(*args, **kwargs):
    return nn.ReLU6(*args, **kwargs)


@ACTIVATION.register()
def Sigmoid(*args, **kwargs):
    return nn.Sigmoid(*args, **kwargs)


@ACTIVATION.register()
def LeakyReLU(*args, **kwargs):
    return nn.LeakyReLU(*args, **kwargs)


@ACTIVATION.register()
def Tanh(*args, **kwargs):
    return nn.Tanh(*args, **kwargs)


@ACTIVATION.register()
def PReLU(*args, **kwargs):
    return nn.PReLU(*args, **kwargs)
