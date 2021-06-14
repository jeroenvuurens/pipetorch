import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
import timeit
import sys
import copy
import inspect
import numpy as np
import math
from tqdm.notebook import tqdm
from ..evaluate.evaluate import Evaluator
from torch.optim.lr_scheduler import OneCycleLR, LambdaLR
from .tuner import *
from functools import partial
import os
try:
    GPU = int(os.environ['GPU'])
    GPU = 0
except:
    GPU = -1
    
def last_container(last):
    try:
        l = last_container(last.children())
        if l is not None:
            return l
    except: pass
    try:
        if len(last._modules) > 0 and next(reversed(last._modules.values())).out_features > 0:
            return last
    except: pass

def to_numpy(arr):
    try:
        return arr.data.cpu().numpy()
    except: pass
    try:
        return arr.to_numpy()
    except: pass
    return arr

class DLModel(nn.Module):
    def __init__(self):
        super().__init__()

    def set_last_linear(self, out_features):
        container = self.last_container()
        name, last = container._modules.popitem()
        container.add_module(name, nn.Linear(last.in_features, out_features))

    def last_container(self):
        return last_container(self)

def UniformLR(*args, **kwargs):
    class Uniform_Scheduler:
        def step(self):
            pass
    return Uniform_Scheduler()

def onecycle(optimizer, lr, steps):
    return OneCycleLR(optimizer, lr[1], total_steps=steps)

class ordered_dl:
    def __init__(self, dl):
        self.dl = dl

    def __enter__(self):
        self.oldsampler = self.dl.batch_sampler.sampler
        self.newsampler = torch.utils.data.sampler.SequentialSampler(self.oldsampler.data_source)
        self.dl.batch_sampler.sampler = self.newsampler
        return self.dl

    def __exit__(self, exc_type, exc_value, tb):
        self.dl.batch_sampler.sampler = self.oldsampler
        if exc_type is not None:
            return False

class trainer:
    def __init__(self, model, loss, *data, report_frequency=1, report_phases=['train','valid'], metrics = [], optimizer=AdamW, optimizerparams=dict(), out_features=None, random_state=None, cycle_epochs=1.0, scheduler=None, weight_decay=None, momentum=None, device=None, gpu=None, evaluator=None, **kwargs):
        self.report_frequency = report_frequency
        self.report_phases = report_phases
        self.loss = loss
        self.random_state = random_state
        self.cycle_epochs = cycle_epochs
        if gpu is not None:
            if gpu == -1:
                device = torch.device('cpu')
            else:
                device = torch.device(f'cuda:{gpu}')
        self.device = device
        self.set_data(*data)
        self._model = model
        try:
            self.post_forward = model.post_forward
        except: pass
        if out_features is not None:
            self._out_features = out_features
        self._optimizerclass = optimizer
        self._optimizerparams = optimizerparams
        self.scheduler = scheduler
        if self.random_state is not None:
            torch.backends.cudnn.deterministic=True
            torch.manual_seed(self.random_state)
        self._commit = {}
        self.epochid = 0
        self.weight_decay = weight_decay
        self.momentum = momentum
        self.lowest_score=None
        self.highest_score=None
        if evaluator is not None:
            assert len(metrics) == 0, 'When you assign an evaluator, you cannot assign different metrics to a trainer'
            self._evaluator = evaluator
            self.metrics = evaluator.metrics
        else:
            self.metrics = metrics

    def set_data(self, *data):
        assert len(data) > 0, 'You have to specify a data source. Either a databunch or a set of dataloaders'
        if len(data) == 1:
            db = data[0]
            self.data = db
        elif len(data) < 4:
            try:
                _ = iter(data[0])
                self.train_dl = data[0]
            except TypeError:
                raise TypeError('The first data source must be iterable, preferably a DataLoader that provide an X and y')
            try:
                _ = iter(data[1])
                self.valid_dl = data[1]
            except TypeError:
                raise TypeError('The second data source must be iterable, preferably a DataLoader that provide an X and y')
            if len(data) > 2:
                try:
                    _ = iter(data[2])
                    self.test_dl = data[2]
                except TypeError:
                    raise TypeError('The third data source must be iterable, preferably a DataLoader that provide an X and y')

    @property
    def evaluator(self):
        try:
            return self._evaluator
        except:
            try:
                self._evaluator = self.db.to_evaluator( *self.metrics )
            except:
                self._evaluator = Evaluator(self, *self.metrics)
            return self._evaluator
            
    def __repr__(self):
        return 'Trainer( ' + self.model + ')'

    def to(self, device):
        self.device = device
        try:
            del self._optimizer
        except: pass

    def cpu(self):
        self.to(torch.device('cpu'))

    def gpu(self):
        self.to(torch.device('cuda:0'))

    @property
    def metrics(self):
        return self._metrics
    
    @metrics.setter
    def metrics(self, value):
        try:
            iter(value)
            self._metrics = value
        except:
            self._metrics = [] if value is None else [value] 
        
    @property
    def data(self):
        return self._data

    @data.setter
    def data(self, db):
        assert hasattr(db, 'train_dl'), 'A single data source must be an object with a train_dl property (like a databunch)'
        assert hasattr(db, 'valid_dl'), 'A single data source must be an object with a valid_dl property (like a databunch)'
        self._data = db
        self.train_dl = self.data.train_dl
        self.valid_dl = self.data.valid_dl
        try:
            self.test_dl = self.data.test_dl
        except: pass

    @property
    def lr(self):
        """
        return: the learning rate that was set, could be an interval
        """
        return self._lr
        
    @lr.setter
    def lr(self, lr):
        """
        Sets the learning rate that is used for training. You can either use a single value
        for a fixed lr, a tuple with an interval of two values for a linear annealing 
        scheduler, or a tuple with an interval of two values for a OneCyleLR scheduler.
        The allocation of a scheduler can be overruled by setting a scheduler manually.
        
        If the lr did not change, nothing happens, otherwise a new optimizer is created
        when needed.
        """
        if type(lr) is tuple:
            lr = tuple(sorted(lr))
        elif type(lr) is list:
            lr = sorted(lr)
        try:
            if self.lr == lr:
                return
        except: pass
        try:
            del self._optimizer
        except: pass
        self._lr = lr

    def set_lr(self, lr):
        """
        sets the learning rate without changing the learning rate settings
        the scheduler or optimizer. is used by tuners like find_lr.
        """
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr


    @property
    def min_lr(self):
        """
        the learning rate or lowest of an interval of learning rates
        """
        try:
            return self.lr[0]
        except:
            try:
                return self.lr
            except:
                return 1e-2

    @property
    def max_lr(self):
        """
        the learning rate or highest of an interval of learning rates
        """
        try:
            return self.lr[1]
        except: pass
        try:
            return self.lr[0]
        except: pass
        return self.lr

    def set_optimizer_param(self, key, value):
        if value is not None:
            self._optimizerparams[key] = value
        else:
            try:
                del self._optimizerparams[key]
            except: pass
        try:
            del self._optimizer
            del self._scheduler
        except: pass

    @property
    def weight_decay(self):
        return self.optimizer.param_groups[0]['weight_decay']

    @weight_decay.setter
    def weight_decay(self, value):
        self.set_optimizer_param('weight_decay', value)

    @property
    def momentum(self):
        return self.optimizer.param_groups[0]['betas']

    @momentum.setter
    def momentum(self, value):
        self.set_optimizer_param('betas', value)

    @property
    def optimizer(self):
        try:
            return self._optimizer
        except:
            self.set_optimizer_param('lr', self.min_lr)
            self._optimizer = self._optimizerclass(self.model.parameters(), **self._optimizerparams)
            return self._optimizer

    @optimizer.setter
    def optimizer(self, value):
        self._optimizerclass = value
        try:
            del self._optimizer
            del self._scheduler
        except: pass

    def del_optimizer(self):
        try:
            del self._optimizer
        except: pass
        try:
            del self._scheduler
        except: pass

    @property
    def scheduler(self):
        """
        Returns: scheduler that is used to adapt the learning rate

        You can either provide a (partial) function that accepts (optimizer, lr, epochs, batch_steps)
        that returns a new scheduler. If you have not specified a function, one of three standard
        schedulers is used based on the value of the learning rate. If the learning rate is a
        single value, the learning rate is fixed. If the learning rate is an interval of two values
        in a list, a linear annealing scheduler is used. If the learning rate is an interval 
        of two values in a tuple, a OneCyleLR scheduler is used.
        """
        try:
            return self._scheduler
        except:
            try:
                #steps = int(round((len(self.train_dl) * self.cycle_epochs)))
                if self.schedulertype is None:
                    try:
                        self.lr[1]
                        if type(self.lr) == tuple:
                            schedulertype = OneCycleLR
                        elif type(self.lr) == list:
                            schedulertype = LambdaLR
                        else:
                            raise NotImplementedError(f'Provide either an single value learning rate for a Uniform scheduler, list [low, high] for a Linear Annealing, or tuple (low, high) for a OneCycleLR scheduler')
                    except:
                        schedulertype = UniformLR
                else:
                    schedulertype = self.schedulertype
                if schedulertype == LambdaLR:
                    if self.cycle_epochs < 2:
                        lr = lambda epoch: self.lr[1]
                    else:
                        lr = lambda epoch: self.lr[1] - (self.lr[1] - self.lr[0]) * epoch / (self.cycle_epochs - 1)
                    self._scheduler = LambdaLR(self.optimizer, lr_lambda = lr)
                elif schedulertype == OneCycleLR:
                    self._scheduler = OneCycleLR(self.optimizer, self.lr[1], epochs=int(self.cycle_epochs), steps_per_epoch=len(self.train_dl))
                else:
                    self._scheduler = schedulertype(self.optimizer, self.lr, self.cycle_epochs, len(self.train_dl))
            except:
                raise NotImplementedError(f'The provided function does not work with (optim, {self.lr}, {self.cycle_epochs}, {len(self.train_dl)}) to instantiate a scheduler')
            return self._scheduler
    @scheduler.setter
    def scheduler(self, value):
        try:
            del self._scheduler
        except: pass
        self.schedulertype = value

    @property
    def out_features(self):
        try:
            return self._out_features
        except: pass
        try:
            self._out_features = last_container(self.model).out_features
            return self._out_features
        except:
            print('cannot infer out_features from the model, please specify it in the constructor of the trainer')
            raise

    @property
    def in_features(self):
        first = next(iter(self._model.modules()))
        while type(first) is nn.Sequential:
            first = next(iter(first.modules()))
        return first.in_features
    
    @property
    def valid_ds(self):
        return self.valid_dl.dataset

    @property
    def train_ds(self):
        return self.train_dl.dataset

    @property
    def test_ds(self):
        return self.test_dl.dataset

    @property
    def train_Xy(self):
        for batch in self.train_dl:
            yield [ t.to(self.model.device) for t in batch ]
    
    @property
    def valid_Xy(self):
        for batch in self.valid_dl:
            yield [ t.to(self.model.device) for t in batch ]
    
    @property
    def test_Xy(self):
        for batch in self.test_dl:
            yield [ t.to(self.model.device) for t in batch ]
    
    @property
    def valid_tensors(self):
        return self.valid_dl.dataset.tensors

    @property
    def train_tensors(self):
        return self.train_dl.dataset.tensors

    @property
    def test_tensors(self):
        return self.test_dl.dataset.tensors

    @property
    def train_X(self):
        return self.train_tensors[0]

    @property
    def train_y(self):
        return self.train_tensors[-1]

    @property
    def valid_X(self):
        return self.valid_tensors[0]

    @property
    def valid_y(self):
        return self.valid_tensors[-1]

    @property
    def test_X(self):
        return self.test_tensors[0]

    @property
    def test_y(self):
        return self.test_tensors[-1]
    
    @property
    def model(self):
        try:
            if self.device is not self._model.device:
                self._model.device = self.device
                self._model.to(self.device)
                try:
                    del self._optimizer
                except: pass
        except:
            try:
                self._model.device = self.device
                self._model.to(self.device)
                #print('change device')
                try:
                    del self._optimizer
                except: pass
            except: pass
        return self._model

    def parameters(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                print(name, param.data)

    def predict(self, *X):
        with self.eval_mode:
            X = [ x.to(self.model.device) for x in X ]
            return self.post_forward(self.model(*X))

    def predict_numpy(self, *X):
        """
        This will only work when instantiated with a PipeTorch Databunch.
        
        """
        self.data
        
        with self.eval_mode:
            X = [ x.to(self.model.device) for x in X ]
            return self.post_forward(self.model(*X))


    def post_forward(self, y):
        return y

    def list_commits(self):
        return self._commit.keys()

    def commit(self, label):
        "save the model and optimzer state, allowing to revert to a previous state"
        model_state = copy.deepcopy(self.model.state_dict())
        optimizer_state = copy.deepcopy(self.optimizer.state_dict())
        self._commit[label] = (model_state, optimizer_state, self._optimizerparams)

    def revert(self, label):
        "revert the model and optimizer to a previously commited state, deletes the commit point"
        if label in self._commit:
            model_state, optimizer_state, self._optimizerparams = self._commit.pop(label)
            self.model.load_state_dict(model_state)
            self.del_optimizer()            
            self.optimizer.load_state_dict(optimizer_state)
        else:
            print('commit point {label} not found')
    
    def checkout(self, label):
        "switches the model and optimizer to a previously commited state, keeps the commit point"
        if label in self._commit:
            model_state, optimizer_state, self._optimizerparams = self._commit[label]
            self.model.load_state_dict(model_state)
            self.del_optimizer()            
            self.optimizer.load_state_dict(optimizer_state)  
        else:
            print('commit point {label} not found')

    def remove_checkpoint(self, label):
        self._commit.pop(label)

    def purge(self, label):
        "switches the model and optimizer to a previously commited state, keeps only the commit point"
        if label in self._commit:
            self.checkout(label)
            self._commit = { l:s for l, s in self._commit.items() if l == label }
        else:
            print(f'commit point {label} not found')

    def validate_loss(self, dl=None):
        if not dl:
            dl = self.valid_Xy
        with self.eval_mode:
            losses = []
            for *X, y in dl:
                losses.append((self.loss_xy(*X, y=y)[0].item() * len(y), len(y)))
            sums = [ sum(x) for x in zip(*losses) ]
            return sums[0] / sums[1]

    @property
    def eval_mode(self):
        class CM(object):
            def __init__(self, trainer):
                self.trainer = trainer
            def __enter__(self):
                self.trainer.model.eval()
                self.prev = torch.is_grad_enabled()
                torch.set_grad_enabled(False)
                return self.trainer.model
            def __exit__(self, type, value, traceback):
                torch.set_grad_enabled(self.prev)
                self.trainer.model.train()
        return CM(self)

    @property
    def train_mode(self):
        class CM(object):
            def __init__(self, trainer):
                self.trainer = trainer
            def __enter__(self):
                self.trainer.model.train()
                self.prev = torch.is_grad_enabled()
                torch.set_grad_enabled(True)
                return self.trainer.model
            def __exit__(self, type, value, traceback):
                torch.set_grad_enabled(self.prev)
                self.trainer.model.eval()
        return CM(self)

    def validate(self, pbar=None, log={}):
        epochloss = 0
        n = 0
        epoch_y_pred = []
        epoch_y = []

        with self.eval_mode:
            for *X, y in self.valid_Xy:
                loss, y_pred = self.loss_xy(*X, y=y)
                epochloss += loss.item() * len(y_pred)
                n += len(y_pred)
                epoch_y_pred.append(to_numpy(y_pred))
                epoch_y.append(to_numpy(y))
                if pbar is not None:
                    pbar.update(self.valid_dl.batch_size)
            epochloss /= n
            epoch_y = np.concatenate(epoch_y, axis=0)
            epoch_y_pred = np.concatenate(epoch_y_pred, axis=0)
            self.evaluator._store(epoch_y, epoch_y_pred, loss=epochloss, phase='valid', epoch=self.epochid, **log)
        return epochloss
            
    def loss_xy(self, *X, y=None):
        assert y is not None, 'Call loss_xy with y=y'
        y_pred = self.model(*X)
        return self.loss(y_pred, y), self.post_forward(y_pred)

    def train_batch(self, *X, y=None):
        self.optimizer.zero_grad()
        loss, y_pred = self.loss_xy(*X, y=y)
        loss.backward()
        self.optimizer.step()
        return loss, y_pred
        
    def _time(self):
        try:
            t = self._start_time
        except:
            t = timeit.default_timer()
        self._start_time = timeit.default_timer()
        return timeit.default_timer() - t
    
    def train(self, epochs, lr=None, report_frequency=None, save=None, optimizer=None, scheduler=False, weight_decay=None, momentum=None, save_lowest=None, save_highest=None, log={}):
        self.del_optimizer()
        self.lr = lr or self.lr
        if weight_decay is not None and self.weight_decay != weight_decay:
            self.weight_decay = weight_decay
        if momentum is not None and self.momentum != momentum:
            self.momentum = momentum
        if optimizer and self._optimizerclass != optimizer:
            self.optimizer = optimizer
        if scheduler is not False:
            self.scheduler = scheduler
        self.report_frequency = report_frequency or self.report_frequency
        model = self.model
        torch.set_grad_enabled(False)
        reports = math.ceil(epochs / self.report_frequency)
        maxepoch = self.epochid + epochs
        epochspaces = int(math.log(maxepoch)/math.log(10)) + 1
        batches = len(self.train_dl) * self.train_dl.batch_size * epochs + len(self.valid_dl) * self.valid_dl.batch_size * reports
        pbar = tqdm(range(batches), desc='Total', leave=False)
        self._time()
        for i in range(epochs):
            self.epochid += 1
            epochloss = 0
            n = 0
            epoch_y_pred = []
            epoch_y = []
            try:
                del self._scheduler
            except: pass
            self.scheduler
            report = (((i + 1) % self.report_frequency) == 0 or i == epochs - 1)
            with self.train_mode:
                for *X, y in self.train_Xy:
                    loss, y_pred = self.train_batch(*X, y=y)
                    self.scheduler.step()
                    try:
                        # TODO naam aanpassen
                        y_pred = model.post_forward(y_pred)
                    except: pass
                    if report:
                        epochloss += loss.item() * len(y_pred)
                        n += len(y_pred)
                        epoch_y_pred.append(to_numpy(y_pred))
                        epoch_y.append(to_numpy(y))

                    pbar.update(self.train_dl.batch_size)
            if report:
                epochloss /= n
                epoch_y = np.concatenate(epoch_y, axis=0)
                epoch_y_pred = np.concatenate(epoch_y_pred, axis=0)
                self.evaluator._store(epoch_y, epoch_y_pred, loss=epochloss, phase='train', epoch=self.epochid, **log)
                validloss = self.validate(pbar = pbar, log=log)
                metric = ''
                v = self.evaluator.valid.iloc[-1]
                for m in self.metrics:
                    m = m.__name__
                    value = v[m]
                    metric += f'{m}={value:.5f} '
                print(f'{self.epochid:>{epochspaces}} {self._time():.2f}s trainloss={epochloss:.5f} validloss={validloss:.5f} {metric}')
                if save is not None:
                    self.commit(f'{save}-{self.epochid}')
                if save_lowest is not None:
                    if self.lowest_score is None or validloss < self.lowest_score:
                        self.lowest_score = validloss
                        self.commit('lowest')
                if save_highest is not None:
                    if self.highest_score is None or validloss > self.highest_score:
                        self.highest_score = validloss
                        self.commit('highest')
    
    def lowest(self):
        self.checkout('lowest')

    def highest(self):
        self.checkout('highest')

    def learning_curve(self, y='loss', series='phase', select=None, xlabel = None, ylabel = None, title=None, label_prefix='', **kwargs):
        return self.evaluator.line_metric(x='epoch', series=series, select=select, y=y, xlabel = xlabel, ylabel = ylabel, title=title, label_prefix=label_prefix, **kwargs)
        
    def validation_curve(self, y=None, x='epoch', series='phase', select=None, xlabel = None, ylabel = None, title=None, label_prefix='', **kwargs):
        if y is not None and type(y) != str:
            y = y.__name__
        return self.evaluator.line_metric(x=x, series=series, select=select, y=y, xlabel = xlabel, ylabel = ylabel, title=title, label_prefix=label_prefix, **kwargs)
       
    def freeze(self, last=-1):
        for c in list(self.model.children())[:last]:
            for p in c.parameters():
                p.requires_grad=False

    def unfreeze(self):
        for c in list(self.model.children()):
            for p in c.parameters():
                p.requires_grad=True

    def tune(self, params,setter, lr=[1e-6, 1e-2], steps=40, smooth=0.05, label=None, **kwargs):
        lr_values = exprange(*lr, steps)
        if label is None:
            label = str(setter)
        if len(params) == 2:
            params = range3(*params)
        with tuner(self, lr_values, self.set_lr, smooth=0.05, label=label) as t:
            t.run_multi(params, setter)

    def tune_weight_decay(self, lr=[1e-6,1e-4], params=[1e-6, 1], steps=40, smooth=0.05, yscale='log', **kwargs):
        self.tune( params, partial(self.set_optimizer_param, 'weight_decay'), lr=lr, steps=steps, smooth=smooth, label='weight decay', yscale=yscale, **kwargs)

    def lr_find(self, lr=[1e-6, 10], steps=40, smooth=0.05, cache_valid=True, **kwargs):
        with tuner(self, exprange(lr[0], lr[1], steps), self.set_lr, label='lr', yscale='log', smooth=smooth, cache_valid=cache_valid, **kwargs) as t:
            t.run()