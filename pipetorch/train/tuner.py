from __future__ import print_function, with_statement, division
import torch
from tqdm.notebook import tqdm
#from torch.optim.lr_scheduler import _LRScheduler
import matplotlib.pyplot as plt
#from .train_metrics import loss
import numpy as np
from .helper import Plot
import sys
from math import log, exp
import statistics
from functools import partial

def frange(start, end, steps):
    incr = (end - start) / (steps)
    return (start + x * incr for x in range(steps))

def exprange(start, end, steps, **kwargs):
    return (exp(x) for x in frange(log(start), log(end), steps))

def arange(start, end, steps, **kwargs):
    return np.arange(start, end, steps)

def set_dropouts(dropouts):
    def change(value):
        for d in dropouts:
            d.p = value
    return change

class tuner:
    def __init__(self, trainer, lrvalues, lrupdate=None, xlabel='parameter', smooth=0.05, diverge=10, max_validation_mem=None, **kwargs):
        self.history = {"lr": [], "loss": []}
        self.best_loss = None
        self.xlabel = xlabel
        self.trainer = trainer
        self.lrvalues = list(lrvalues)
        self.lrupdate = lrupdate if lrupdate else trainer.set_lr
        self.smooth = smooth
        self.diverge = diverge
        self.max_validation_mem = max_validation_mem

    def __enter__(self):
        self.trainer.commit('tuner')
        return self

    def __exit__(self, *args):
        self.trainer.revert('tuner')

    def next_train(self):
        try:
            return next(self.train_Xy)
        except (StopIteration, AttributeError):
            self.train_iterator = iter(self.trainer.train_Xy)
            return next(self.train_iterator)

    def run( self, cache_valid=True ):
        graphx = []
        sloss = []
        validation_set = []
        mem_validation = 0
        self.trainer.model
        
        if cache_valid:
            for batch in self.trainer.valid_Xy:
                validation_set.append(batch)
                mem_validation += sum([sys.getsizeof(x.storage()) for x in batch])
                #print(mem_validation)
                if self.max_validation_mem and mem_validation > self.max_validation_mem:
                    print('warning: validation set is too large for memory')
                    break
        else:
            validation_set = self.trainer.valid_Xy
        #with plt_notebook():
        with Plot(xscale='log', xlabel=self.xlabel) as p:
            with self.trainer.train_mode:
                for i, lr in enumerate(tqdm(self.lrvalues, leave=False)):
                    graphx.append(lr)
                    self.lrupdate(lr)
                    *X, y = self.next_train()
                    loss, pred_y = self.trainer.train_batch(*X, y=y)
                    loss = self.trainer.loss_dl(validation_set)
                    try:
                        loss = self.smooth * loss + (1 - self.smooth) * sloss[-1]
                    except: pass
                    sloss.append(loss)

                    min_index = np.argmin(sloss) + 1
                    maxx = max(sloss[:min_index])
                    minn = min(sloss[:min_index])
                    if i > len(self.lrvalues) / 4 and loss > maxx * 1.1 and loss > maxx:
                        #print("Stopping early, the loss has diverged")
                        break
                    p.replot( graphx, sloss )

    def run_multi( self, param2_values, param2_update ):
        param2_values = list(param2_values)
        for p in param2_values:
            param2_update(p)
            self.trainer.commit(f'param2_{p:.2E}')
        x = []
        sloss = { f'{p:.2E}':[] for p in param2_values }
        #with plt_notebook():
        with Plot(xscale='log', xlabel=self.xlabel) as plot:

            dropped_param2_values = []
            for lr in tqdm(self.lrvalues, leave=False):
                with self.trainer.train_mode:
                    x.append(lr)
                    *X, y = self.next_train()
                    for p in param2_values:
                        self.trainer.checkout(f'param2_{p:.2E}')
                        param2_update(p)
                        self.lrupdate(lr)
                        loss, pred_y = self.trainer.train_batch(*X, y=y)
                        loss = self.trainer.validate_loss()
                        try:
                            loss = smooth * loss + (1 - smooth) * sloss[f'{p:.2E}'][-1]
                        except: pass
                        sloss[f'{p:.2E}'].append(loss)
                        #print(self.trainer.optimizer.param_groups[0]['weight_decay'])
                        #print(f'param2_{p:.2E} {loss}')

                        try:
                            if loss > diverge * min_loss:
                                dropped_param2_values.append(p)
                            min_loss = min(min_loss, loss)
                        except:
                            min_loss = loss
                    for p in param2_values:
                        self.trainer.commit(f'param2_{p:.2E}')
                    plot.multiplot( x, sloss )

        for p in param2_values:
            self.trainer.remove_checkpoint(f'param2_{p:.2E}')

