import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import timeit
import numpy as np
#from .train_diagnostics2 import *
from .train_diagnostics import *
from .train_metrics import *
from .jcollections import *
from functools import partial

class train_history:
    def __init__(self, trainer):
        self.trainer = trainer
        self.epochs = defaultdict(list)
        self.added_modules = set()
        self.before_batch = []
        self.before_epoch = []
        self.after_epoch = []
        self.after_batch = []
        self.metrics
#        for m in self.trainer.modules:
#            self.require_module(m)
       
    @property
    def metrics(self):
        try: 
            return self._metrics
        except AttributeError:
            self._metrics = [ m(self) for m in self.trainer.metrics ]
#            for m in self._metrics:
#                m.requirements()
            return self._metrics

#    def create_modules(self):
#        self.require_module(self.trainer.modules)

#    def require_module(self, *modules):
#        for module in modules:
#            if module.__name__ not in self.added_modules:
#                m = module(self)
#                if m.requirements():
#                    print(f"not using module {module.__name__}")
#                else:
#                    #print(f"adding module {module.__name__}")
#                    if getattr(m, "before_batch", None) != None:
#                        self.before_batch.append(m)
#                    if getattr(m, "after_batch", None) != None:
#                        self.after_batch.append(m)
#                    if getattr(m, "before_epoch", None) != None:
#                        self._before_epoch.append(m)
#                    if getattr(m, "after_epoch", None) != None:
#                        self.after_epoch.append(m)
#                    self.added_modules.add(module.__name__)

    def create_epoch(self, phase):
        return Epoch(self, phase)

    def register_epoch(self, epoch):
        self.epochs[epoch['phase']].append(epoch)

    def train(self, metric, start=0, end=None):
        if end is None:
            end=len(self.epochs['train'])
        return [ metric.value(epoch) for epoch in self.epochs['train'][start:end] ]

    def valid(self, metric, start=0, end=None):
        if end is None:
            end=len(self.epochs['train'])
        return [ metric.value(epoch) for epoch in self.epochs['valid'][start:end] ]

    def plot(self, *metric, **kwargs):
        if len(metric) == 0:
            self.plotf(loss(self), 'loss', **kwargs)
        else:
            for m in metric:
                m = m(self)
                self.plotf(m, m.__name__, **kwargs)

    def plotf(self, metric, ylabel, start=0, end=None, **kwargs):
        if end is None:
            end=len(self.epochs['train'])
        x = [ epoch['epoch'] for epoch in self.epochs['train'][start:end] ]
        plt.figure(**kwargs)
        plt.plot(x, self.train(metric, start, end), label='train')
        plt.plot(x, self.valid(metric, start, end), label='valid')
        plt.ylabel(ylabel)
        plt.xlabel("epochs")
        plt.legend()
        plt.show()

    def ploti(self, plot, metric):
        x = [ epoch['epoch'] for epoch in self.epochs['train'] ]
        y = { 'train':self.train(metric), 'valid':self.valid(metric) }
        plot.multiplot(x, y)

class Epoch(dict):
    def __init__(self, history, phase):
        super().__init__()
        trainer = history.trainer
        self['epoch'] = trainer.epochid
        self['phase'] = phase
        self.history = history
        self.report = False

    def time(self):
        return self['endtime'] - self['starttime']

    def before_epoch(self):
        if self.report:
            self['starttime'] = timeit.default_timer()
            self.y = []
            self.y_pred = []
            self['loss'] = 0
            self['n'] = 0

    def before_batch(self, X, y):
        pass

    def after_batch(self, X, y, y_pred, loss):
        if self.report:
            self.y.append(y)
            self.y_pred.append(y_pred)
            self['loss'] += loss * len(y)
            self['n'] += len(y)

    def after_epoch(self):
        if self.report:
            self.y = to_numpy(torch.cat(self.y))
            self.y_pred = to_numpy(torch.cat(self.y_pred))
            self['loss'] = to_numpy(self['loss']) / self['n']
            #for m in self.history.after_epoch:
            #    m.after_epoch(self, self.y, self.y_pred, self.loss)
            for m in self.history.metrics:
                self[m.__name__] = m.value(self)
            del self.y
            del self.y_pred
            self['endtime'] = timeit.default_timer()

    def __repr__(self):
        return ( self["phase"] + \
            ' '.join([ f' {metric.__name__}: {metric.value(self):.6f}'
                for metric in self.history.metrics ]))

