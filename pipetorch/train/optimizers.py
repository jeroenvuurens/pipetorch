import torch
from torch.autograd import Variable
import torch.nn as nn
from torch.optim.optimizer import *
from torch.optim.lr_scheduler import *
import timeit
import copy
import numpy as np
import math
from tqdm.notebook import tqdm
#from tqdm import tqdm_notebook as tqdm
#from .train_diagnostics2 import *
from .train_diagnostics import *
from .train_metrics import *
from .train_history import *
from .jcollections import *
#from .transfer import *
from .helper import *
from functools import partial

class CLR(CyclicLR):
    def __init__(self,
                 optimizer,
                 base_lr,
                 max_lr,
                 step_size_up=2000,
                 step_size_down=None,
                 mode='triangular',
                 gamma=1.,
                 scale_fn=None,
                 scale_mode='cycle',
                 cycle_momentum=True,
                 base_momentum=0.8,
                 max_momentum=0.9,
                 last_epoch=-1):

        if not isinstance(optimizer, Optimizer):
            raise TypeError('{} is not an Optimizer'.format(
                type(optimizer).__name__))
        self.optimizer = optimizer

        base_lrs = self._format_param('base_lr', optimizer, base_lr)
        if last_epoch == -1:
            for lr, group in zip(base_lrs, optimizer.param_groups):
                group['lr'] = lr

        self.max_lrs = self._format_param('max_lr', optimizer, max_lr)

        step_size_up = float(step_size_up)
        step_size_down = float(step_size_down) if step_size_down is not None else step_size_up
        self.total_size = step_size_up + step_size_down
        self.step_ratio = step_size_up / self.total_size

        if mode not in ['triangular', 'triangular2', 'exp_range'] \
                and scale_fn is None:
            raise ValueError('mode is invalid and scale_fn is None')

        self.mode = mode
        self.gamma = gamma

        if scale_fn is None:
            if self.mode == 'triangular':
                self.scale_fn = self._triangular_scale_fn
                self.scale_mode = 'cycle'
            elif self.mode == 'triangular2':
                self.scale_fn = self._triangular2_scale_fn
                self.scale_mode = 'cycle'
            elif self.mode == 'exp_range':
                self.scale_fn = self._exp_range_scale_fn
                self.scale_mode = 'iterations'
        else:
            self.scale_fn = scale_fn
            self.scale_mode = scale_mode

        self.cycle_momentum = cycle_momentum
        if cycle_momentum:
            if 'momentum' not in optimizer.defaults:
                raise ValueError('optimizer must support momentum with `cycle_momentum` option enabled')

            base_momentums = self._format_param('base_momentum', optimizer, base_momentum)
            if last_epoch == -1:
                for momentum, group in zip(base_momentums, optimizer.param_groups):
                    group['momentum'] = momentum
            self.base_momentums = list(map(lambda group: group['momentum'], optimizer.param_groups))
            self.max_momentums = self._format_param('max_momentum', optimizer, max_momentum)

        super().__init__(optimizer, base_lr, max_lr, last_epoch=last_epoch, cycle_momentum=cycle_momentum)

def cyclicallr(optimizer, min_lr, max_lr, steps):
    steps_size_down=steps - steps // 2
    return CLR(optimizer, min_lr, max_lr, step_size_up=steps // 2, step_size_down=steps_size_down, mode='triangular', cycle_momentum=False)

def onecyclelr(optimizer, min_lr, max_lr, steps):
    return OneCycleLR(optimizer, max_lr=max_lr, total_steps=steps)

def uniformlr():
    class Uniform_Scheduler:
        def step(self):
            pass
    return Uniform_Scheduler()

