import torch.optim as optim
import sys
import os
from pathlib import Path

FILE = Path(__file__).resolve()
ROOT = FILE.parents[2]  # root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative
RANK = int(os.getenv('RANK', -1))

from core.builder import OPTIMIZER


@OPTIMIZER.register()
def SGD(*args, **kwargs):
    return optim.SGD(*args, **kwargs)


@OPTIMIZER.register()
def Adam(*args, **kwargs):
    return optim.Adam(*args, **kwargs)


@OPTIMIZER.register()
def SparseAdam(*args, **kwargs):
    return optim.SparseAdam(*args, **kwargs)


@OPTIMIZER.register()
def AdamW(*args, **kwargs):
    return AdamW(*args, **kwargs)


@OPTIMIZER.register()
def Adadelta(*args, **kwargs):
    return optim.Adadelta(*args, **kwargs)


@OPTIMIZER.register()
def ASGD(*args, **kwargs):
    return optim.ASGD(*args, **kwargs)


@OPTIMIZER.register()
def RMSprop(*args, **kwargs):
    return optim.RMSprop(*args, **kwargs)


@OPTIMIZER.register()
def Rprop(*args, **kwargs):
    return optim.Rprop(*args, **kwargs)


@OPTIMIZER.register()
def RAdam(*args, **kwargs):
    return optim.RAdam(*args, **kwargs)


@OPTIMIZER.register()
def NAdam(*args, **kwargs):
    return optim.NAdam(*args, **kwargs)


@OPTIMIZER.register()
def LBFGS(*args, **kwargs):
    return optim.LBFGS(*args, **kwargs)


@OPTIMIZER.register()
def Adamax(*args, **kwargs):
    return optim.Adamax(*args, **kwargs)
