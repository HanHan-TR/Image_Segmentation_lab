from typing import Optional, Union
import torch.optim.lr_scheduler as lr_scheduler
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler  # , StepLR, PolynomialLR, ExponentialLR, ChainedScheduler, LambdaLR

import sys
import os
from pathlib import Path

FILE = Path(__file__).resolve()
ROOT = FILE.parents[2]  # root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative
RANK = int(os.getenv('RANK', -1))

from core.registry import LR_SCHEDULER


@LR_SCHEDULER.register()
def StepLR(*args, **kwargs):
    return lr_scheduler.StepLR(*args, **kwargs)


@LR_SCHEDULER.register()
def PolynomialLR(*args, **kwargs):
    return lr_scheduler.PolynomialLR(*args, **kwargs)


@LR_SCHEDULER.register()
def ExponentialLR(*args, **kwargs):
    return lr_scheduler.ExponentialLR(*args, **kwargs)


@LR_SCHEDULER.register()
def LambdaLR(*args, **kwargs):
    return lr_scheduler.LambdaLR(*args, **kwargs)


class WarmScheduler(_LRScheduler):
    def __init__(self,
                 optimizer: Optimizer,
                 warmup_iters: int = 50,
                 mode: str = 'linear',
                 start_lr: Optional[float] = None,
                 start_ratio: Optional[float] = 0.1,
                 end_lr: Optional[float] = None,
                 last_epoch: int = -1):
        self.warmup_iters = warmup_iters
        self.mode = mode

        if start_lr is None and start_ratio is not None:
            self.warmup_start_lr = optimizer.param_groups[0]['lr'] * start_ratio
        else:
            self.warmup_start_lr = start_lr

        if end_lr is None:
            self.warmup_end_lr = optimizer.param_groups[0]['lr']
        else:
            self.warmup_end_lr = end_lr

        self.finished_warmup = False  # warmup是否完成的标志
        super(WarmScheduler, self).__init__(optimizer, last_epoch)

    def get_lr(self, iters: int):
        if iters < self.warmup_iters:
            if self.mode == 'linear':
                warmup_lr = (self.warmup_end_lr - self.warmup_start_lr) * iters / self.warmup_iters + self.warmup_start_lr
            elif self.mode == 'exponential':
                warmup_lr = self.warmup_start_lr * (self.warmup_end_lr / self.warmup_start_lr) ** (iters / self.warmup_iters)
            else:
                raise ValueError(f"Unsupported warmup mode: {self.mode}")
            return [warmup_lr for _ in self.base_lrs]
        else:
            if not self.finished_warmup:
                self.finished_warmup = True
                print(f"Warmup finished, start training with lr={self.base_lrs}")

            return [base_lr for base_lr in self.base_lrs]
