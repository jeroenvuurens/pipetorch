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
from torch.optim.lr_scheduler import OneCycleLR, ConstantLR
from .tuner import *
from .helper import nonondict
from functools import partial
import os
try:
    GPU = int(os.environ['GPU'])
    GPU = 0
except:
    GPU = -1
    
# def last_container(last):
#     try:
#         l = last_container(last.children())
#         if l is not None:
#             return l
#     except: pass
#     try:
#         if len(last._modules) > 0 and next(reversed(last._modules.values())).out_features > 0:
#             return last
#     except: pass

def to_numpy(arr):
    try:
        return arr.data.cpu().numpy()
    except: pass
    try:
        return arr.to_numpy()
    except: pass
    return arr

# class DLModel(nn.Module):
#     def __init__(self):
#         super().__init__()

#     def set_last_linear(self, out_features):
#         container = self.last_container()
#         name, last = container._modules.popitem()
#         container.add_module(name, nn.Linear(last.in_features, out_features))

#     def last_container(self):
#         return last_container(self)

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
    """
    A general purpose trainer for PyTorch.
    
    Arguments:
        model: nn.Module
            a PyTorch Module that will be trained
            
        loss: callable
            a PyTorch or custom loss function
            
        data: databunch or a list of iterables (DataLoaders)
            a databunch is an object that has a train_dl, valid_dl,
            and optionally test_dl property.
            otherwise, a list of iterables can also be given. 
            Most often, these iterables are PyTorch DataLoaders that 
            are used to iterate over the respective datasets
            for training and validation.
            
        metrics: callable or list of callable
            One or more functions that can be called with (y, y_pred)
            to compute an evaluation metric. This will automatically be
            done during training, for both the train and valid sets.
            Typically, the callable is a function from SKLearn.metrics
            like mean_squared_error or recall_score.
            
        optimizer: PyTorch Optimizer (AdamW)
            The PyTorch or custom optimizer class that is used during training
            
        optimizerparams: dict (None)
            the parameters that are passed (along with the model parameters)
            to initialize an optimizer. A 'nonondict' is used, meaning that
            when a None value is set, the key is removed, so that the default
            value is used instead.
            
        random_state: int
            used to set a random state for reproducible results
            
        scheduler: None, OneCycleLR, ConstantLR
            used to adapt the learning rate: 
            - None will use a constant learning rate
            - OneCycleLR will will use a cyclic annealing learning rate
              between an upper and lower bound.
            - ConstantLR will use a linear decaying learning rate between
              an upper bound and lower bound. You can optionally use
              'cycle' when calling 'train' to restart ConstantLR 
              every 'cycle' epochs.
              
        weight_decay: float
            Apply weight_decay regularization with the AdamW optimizer
            
        momentum: float
            Apply momentum with the AdamW optimizer
            
        gpu: bool, int or torch.device
            The device to train on:
                False or -1: cpu
                True: cuda:0, this is probably what you want to train on gpu
                int: cuda:gpu
            Setting the device will automatically move the model and data to
            the given device. Note that the model is not automatically
            transfered back to cpu afterwards.
    
        evaluator: PipeTorch evaluator
            An evaluator that was created by a different trainer or 
            DataFrame, to combine the results of different training
            sessions.
    """
    
    def __init__(self, model, loss, *data, metrics = [], optimizer=AdamW, optimizerparams=None, random_state=None, scheduler=None, weight_decay=None, momentum=None, gpu=False, evaluator=None, **kwargs):
        self.report_frequency = 1
        self.loss = loss
        self.random_state = random_state
        self.gpu(gpu)
        self.set_data(*data)
        self._model = model
        try:
            self.post_forward = model.post_forward
        except: pass
        self.optimizer = optimizer
        self.optimizer_params = optimizerparams
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
        """
        Changes the dataset that is used by the trainer
        
        Arguments:
            data: databunch or a list of iterables (DataLoaders)
                a databunch is an object that has a train_dl, valid_dl,
                and optionally test_dl property.
                otherwise, a list of iterables can also be given. 
                Most often, these iterables are PyTorch DataLoaders that 
                are used to iterate over the respective datasets
                for training and validation.
        """
        assert len(data) > 0, 'You have to specify a data source. Either a databunch or a set of dataloaders'
        if len(data) == 1:
            db = data[0]
            self.databunch = db
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
        """
        The (PipeTorch) evaluator that is used to log training progress
        """
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
        """
        Configures the device to train on
        
        Arguments:
            device: bool, int or torch.device
                The device to train on:
                    False or -1: cpu
                    True: cuda:0, this is probably what you want to train on gpu
                    int: cuda:gpu
                Setting the device will automatically move the model and data to
                the given device. Note that the model is not automatically
                transfered back to cpu afterwards.
        """
        if device is True or (type(device) == int and device == 0):
            device = torch.device('cuda:0')
        elif device is False or (type(device) == int and device == -1):
            device = torch.device('cpu')
        elif type(device) == int:
            assert device < torch.cuda.device_count(), 'Cannot use gpu {device}, note that if a gpu has already been selected it is always renumbered to 0'
            device = torch.device(f'cuda:{device}')
        try:
            if device != self.device:
                self.device = device
                try:
                    del self._optimizer
                except: pass
        except:
            self.device = device

    def cpu(self):
        """
        Configure the trainer to train on cpu
        """
        self.to(False)

    def gpu(self, gpu=True):
        """
        Configure the trainer to train on gpu, see to(device)
        """
        self.to(gpu)

    @property
    def metrics(self):
        """
        Returns: list of metrics that is collected while training
        """
        return self._metrics
    
    @metrics.setter
    def metrics(self, value):
        """
        Sets the metric(s) that are collected while training
        """
        try:
            iter(value)
            self._metrics = value
        except:
            self._metrics = [] if value is None else [value] 
        
    @property
    def databunch(self):
        """
        Returns: the databunch that is used
        
        thows an exception if no databunch has been configured
        """
        return self._databunch

    @databunch.setter
    def databunch(self, db):
        """
        Setter to use a databunch. The databunch object must have at least
        a train_dl and a valid_dl property, and optional a test_dl. These
        are often PyTorch DataLoaders, but can be any iterable over a
        DataSet.
        """
        
        assert hasattr(db, 'train_dl'), 'A single data source must be an object with a train_dl property (like a databunch)'
        assert hasattr(db, 'valid_dl'), 'A single data source must be an object with a valid_dl property (like a databunch)'
        self._databunch = db
        self.train_dl = self.databunch.train_dl
        self.valid_dl = self.databunch.valid_dl
        try:
            self.test_dl = self.databunch.test_dl
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
        for a fixed lr, a tuple with an interval of two values for a linear decaying 
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
        """
        Set a parameter for the optimizer. A 'nonondict' is used, 
        meaning that setting a value to None will cause the default
        to be used.
        
        Argument:
            key: str
                the key to use
                
            value: any
                the value to use. When set to None, the key is removed.
        """
        self.optimizer_params[key] = value
        try:
            del self._optimizer
            del self._scheduler
        except: pass

    @property
    def weight_decay(self):
        """
        Returns: the current value for the weight decay regularization
        
        only works when using an Adam(W) optimizer
        """
        return self.optimizer.param_groups[0]['weight_decay']

    @weight_decay.setter
    def weight_decay(self, value):
        """
        Sets the weight decay regularization on the Adam(W) optimizer
        """
        self.set_optimizer_param('weight_decay', value)

    @property
    def momentum(self):
        """
        Returns the momentum value on the Adam(W) optimizer
        """
        return self.optimizer.param_groups[0]['betas']

    @momentum.setter
    def momentum(self, value):
        """
        Sets the momentum value on the Adam(W) optimizer
        """
        self.set_optimizer_param('betas', value)

    @property
    def optimizer(self):
        """
        Returns: an optimizer for training the model, using the applied
        configuration (e.g. weight_decay, momentum, learning_rate).
        If no optimizer exists, a new one is created using the configured
        optimizerclass (default: AdamW) and settings.
        """
        try:
            return self._optimizer
        except:
            self.set_optimizer_param('lr', self.min_lr)
            self._optimizer = self._optimizer_class(self.model.parameters(), **self.optimizer_params)
            return self._optimizer

    @optimizer.setter
    def optimizer(self, value):
        """
        Sets the optimizer class to use. 
        """
        self._optimizer_class = value
        try:
            del self._optimizer
            del self._scheduler
        except: pass

    @property
    def optimizer_params(self):
        try:
            return self._optimizer_params
        except:
            self._optimizer_params = nonondict()
            return self._optimizer_params
    
    @optimizer_params.setter
    def optimizer_params(self, value):
        """
        Setter for the optimizer parameters used, only applies them if
        the value is set other than None. If you want to remove all
        params, set them to an empty dict.
        
        Arguments:
            value: dict
                conform the optimizer class that is used
        """
        if value is not None:
            assert instanceof(value) == dict, 'you have set optimizer_params to a dict'
            self._optimizer_params = nonondict(value)
        
    @property
    def scheduler_params(self):
        try:
            return self._scheduler_params
        except:
            self._scheduler_params = nonondict()
            return self._scheduler_params
    
    @scheduler_params.setter
    def scheduler_params(self, value):
        """
        Setter for the scheduler parameters used, only applies them if
        the value is set other than None. If you want to remove all
        params, set them to an empty dict.
        
        Arguments:
            value: dict
                conform the scheduler class/initializer that is used
        """
        if value is not None:
            assert instanceof(value) == dict, 'you have set scheduler_params to a dict'
            self._optimizer_params = nonondict(value)
        
    def del_optimizer(self):
        try:
            del self._optimizer
        except: pass
        self.del_scheduler()

    def del_scheduler(self):
        try:
            del self._scheduler
        except: pass

    @property
    def scheduler(self):
        """
        Returns: scheduler that is used to adapt the learning rate

        When you have set a (partial) function to initialze a scheduler, it should accepts
        (optimizer, lr, scheduler_params) as its parameters. Otherwise, one of three standard
        schedulers is used based on the value of the learning rate. If the learning rate is 
        - float: no scheduler is used
        - [max, min]: a linear decaying scheduler is used. 
        - (max, min): a OneCyleLR scheduler is used.
        """
        try:
            return self._scheduler
        except:
            try:
                #steps = int(round((len(self.train_dl) * self.cycle_epochs)))
                if self._scheduler_class is None:
                    try:
                        self.lr[1]
                        if type(self.lr) == tuple:
                            schedulerclass = OneCycleLR
                        elif type(self.lr) == list:
                            schedulerclass = ConstantLR
                        else:
                            raise NotImplementedError(f'Provide either an single value learning rate for a Uniform scheduler, list [low, high] for a Linear Decay, or tuple (low, high) for a OneCycleLR scheduler')
                    except:
                        schedulerclass = UniformLR
                else:
                    schedulerclass = self._scheduler_class
                if schedulerclass == ConstantLR:
                    factor = (self.min_lr / self.max_lr) ** (1 / self._scheduler_epochs)
                    self._scheduler = ConstantLR(self.optimizer, factor,
                                      self._scheduler_epochs, **self.scheduler_params)
                elif schedulerclass == OneCycleLR:
                    scheduler_params = self.scheduler_params
                    scheduler_params['epochs'] = self._scheduler_epochs
                    scheduler_params['steps_per_epoch'] = len(self.train_dl)
                    self._scheduler = OneCycleLR(self.optimizer, 
                                      self.min_lr, **scheduler_params) 
                else:
                    self._scheduler = schedulerclass(self.optimizer, 
                                      self.lr, **self.scheduler_params)
            except:
                raise NotImplementedError(f'The provided function does not work with (optim, {self.lr}, {self._scheduler_epochs}, {len(self.train_dl)}) to instantiate a scheduler')
            return self._scheduler
    
    @scheduler.setter
    def scheduler(self, value):
        """
        Sets the schedulerclass (or function to initialize a scheduler) to use. At this moment,
        there is no uniform way to initialize all PyTorch schedulers. 
        PipeTorch provides easy support for using a scheduler through the learning rate:
        - float: no scheduler is used
        - [max, min]: a linear annealing scheduler is used. 
        - (max, min): a OneCyleLR scheduler is used.
        
        To use another scheduler, set this to a function that accepts
        the following parameters: (optimizer instance, learning rate, **scheduler_params)
        
        The scheduler_params can be supplied when calling train.
        """
        
        try:
            del self._scheduler
        except: pass
        self._scheduler_class = value

#     @property
#     def out_features(self):
#         try:
#             return self._out_features
#         except: pass
#         try:
#             self._out_features = last_container(self.model).out_features
#             return self._out_features
#         except:
#             print('cannot infer out_features from the model, please specify it in the constructor of the trainer')
#             raise

#     @property
#     def in_features(self):
#         first = next(iter(self._model.modules()))
#         while type(first) is nn.Sequential:
#             first = next(iter(first.modules()))
#         return first.in_features
    
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
        """
        When a device is configured to train the model on, the model
        is automatically transferred to the device. A device property
        is set on the model to transfer the data to the same device
        as the model before using.
        
        Returns: the model 
        """
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
        """
        Prints the (trainable) model parameters
        """
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                print(name, param.data)

    def forward(self, *X):
        """
        Returns the results of the model's forward on the given input X.
             
        Arguments:
            *X: tensor or collection of tensors
                the tensor of collection of tensors that is passed to
                the forward of the model. The inputs are automatically 
                transfered to the same device as the model is on.
        
        Returns: tensor
            outputs that are returned by first the forward pass on
            the model.
        """
        X = [ x.to(self.model.device) for x in X ]
        return self.model(*X)
       
    def predict(self, *X):
        """
        Returns model predictions for the given input.
        The difference with forward is that the outputs of the model
        are optionally processed by a post_forward (for classification).
        
        Arguments:
            *X: tensor or collection of tensors
                the tensor of collection of tensors that is passed to
                the forward of the model. The inputs are automatically 
                transfered to the same device as the model is on.
        
        Returns: tensor
            Predictions that are returned by first the forward pass on
            the model and optionally a post_forward for classification
            tasks
        """
        return self.post_forward(self.forward(*X))

    def post_forward(self, y):
        """
        For classification tasks, training may require a different 
        pred_y than the evaluation metrics do. Typically, the predictions
        are logits or an estimated likelihood (e.g. 0.2), while the 
        evaluation function need a class label (e.g. 0 or 1). Using
        PipeTorch, you need to add a post_forward(y) method to your model,
        that will be called on the predictions before they are passed
        to the evaluation functions. 
        
        Returns: tensor
            If the model has a post_forward to convert pred_y to predictions,
            this returns the the results calling post_forward, otherise,
            it will just return pred_y
        """
        post_forward = getattr(self.model, "post_forward", None)
        if callable(post_forward):
            return self.model.post_forward(y)
        return y

    def list_commits(self):
        """
        Returns: a list of the keys of committed (saved) models, during 
        or after training.
        """
        return self._commit.keys()

    def commit(self, label):
        """
        Save the model and optimizer state, allowing to revert to a 
        previous state/version of the model.
        
        Arguments:
            label: str
                The key to save the model under
        """        
        model_state = copy.deepcopy(self.model.state_dict())
        optimizer_state = copy.deepcopy(self.optimizer.state_dict())
        self._commit[label] = (model_state, optimizer_state)

    def _model_filename(self, folder=None, filename=None, extension=None):
        if folder is None:
            folder = '.'
        if filename is not None:
            path = f'{folder}/{filename}'
        else:
            path = f'{folder}/{self.model.__class__.__name__}'
        if '.pyt' not in path:
            if extension is None:
                return f'{path}.pyt{torch.__version__}'
            else:
                return f'{path}.{extension}'
        return path
        
    def save(self, folder=None, filename=None, extension=None):
        """
        Saves a (trained) model to file. This will only save the model parameters. To load the model, you will
        first have to initialize a model with the same configuration, and then use trainer.load(path) to load
        the model from file.
        
        Aruments:
            folder: str (None)
                folder to save the model, default is the current folder
            filename: str (None)
                the basename of the saved file, default is the classname
            extension: str (None)
                the extension of the saved file, default is pyt with the pytorch version name
        """
        path = self._model_filename(folder, filename, extension)
        torch.save(self.model.state_dict(), path)
        print(f'Saved the model as {path}')
        
    def load(self, folder=None, filename=None, extension=None):
        """
        Load a saved (trained) model from file. For this to work, the model for this trainer has to be configured
        in the exact same way as the model that was saved. This will only load the model parameters.
        
        Aruments:
            folder: str (None)
                folder to save the model, default is the current folder
            filename: str (None)
                the basename of the saved file, default is the classname
            extension: str (None)
                the extension of the saved file, default is pyt with the pytorch version name
        """
        self.model.load_state_dict(torch.load(self._model_filename(folder, filename, extension)))
        
    def to_trt(self):
        """
        Converts the (trained) model into a TRT model that can be used on a Jetson
        
        Returns: TRTModule
            The converted model
        """
        from torch2trt import torch2trt
        x = next(iter(self.train_Xy))[0]
        print(x.shape)
        return torch2trt(self.model, [x])
        
    def save_trt(self, folder=None, filename=None, extension='trt'):
        """
        Converts the (trained) model to TRT and saves it.
        
        Aruments:
            folder: str (None)
                folder to save the model, default is the current folder
            filename: str (None)
                the basename of the saved file, default is the classname
            extension: str ('trt')
                the extension of the saved file
        """
        path = self._model_filename(folder, filename, extension)
        torch.save(self.to_trt().state_dict(), path)
        print(f'Saved the TRT model as {path}')
        
    def save_onnx(self, folder=None, filename=None, extension='onnx'):
        """
        Converts the (trained) model to ONNX and saves it.
        
        Aruments:
            folder: str (None)
                folder to save the model, default is the current folder
            filename: str (None)
                the basename of the saved file, default is the classname
            extension: str ('onnx')
                the extension of the saved file
        """
        path = self._model_filename(folder, filename, extension)
        x = next(iter(self.train_Xy))[0][:1]
        torch.onnx.export(self.model, x, path, verbose=True)
        print(f'Saved the ONNX model as {path}')
        
        
    def revert(self, label):
        """
        Revert the model and optimizer to a previously commited state, 
        and deletes the commit point to free memory. Prints a warning
        when the label was not found.
        
        Arguments:
            label: str
                The key under which the model was commited
        """
        if label in self._commit:
            model_state, optimizer_state = self._commit.pop(label)
            self.model.load_state_dict(model_state)
            self.del_optimizer()            
            self.optimizer.load_state_dict(optimizer_state)
        else:
            print('commit point {label} not found')
    
    def checkout(self, label):
        """
        Loads a previously commited state of the model and optimizer 
        but keeps the commit point. Prints a warning
        when the label was not found.
        
        Arguments:
            label: str
                The key under which the model was commited
        """
        if label in self._commit:
            model_state, optimizer_state = self._commit[label]
            self.model.load_state_dict(model_state)
            self.del_optimizer()            
            self.optimizer.load_state_dict(optimizer_state)  
        else:
            print('commit point {label} not found')

    def remove_checkpoint(self, label):
        """
        Removes a previously committed state of the model.
        
        Arguments:
            label: str
                The key under which the model was commited
        """
        self._commit.pop(label)

    def purge(self, label):
        """
        Switches the model and optimizer to a previously commited state, 
        and keeps only that commit point and removes all other versions.
        
        Arguments:
            label: str
                The key under which the model was commited
        """
        if label in self._commit:
            self.checkout(label)
            self._commit = { l:s for l, s in self._commit.items() if l == label }
        else:
            print(f'commit point {label} not found')

    def _loss_xy(self, *X, y=None):
        """
        Computes predictions for the given X.
        
        Arguments:
            *X: tensor
                inputs that are used by the forward of the model
            y: tensor
                ground truth labels, the predictions are compared against
        
        Returns: (float, tensor)
            a tuple with the loss for the predictions on X,
            and a tensor with the predicted values
        """
        assert y is not None, 'Call _loss_xy with y=None'
        y_pred = self.forward(*X)
        return self.loss(y_pred, y), self.post_forward(y_pred)
    
    def loss_dl(self, dl):
        """
        Iterates over the given dataloader, the loss is computed in
        evaluation mode and accumulated over the dataset.
        
        Arguments:
            dl: DataLoader
                the dataloader that is used to iterate over.
        
        Returns: float 
            weighted average loss over the given dataloader/set.
        """
        if not dl:
            dl = self.valid_Xy
        losses = []
        leny = 0
        for *X, y in dl:
            y_pred = self.forward(*X)
            l = self.loss(y_pred, y)
            losses.append(l.item() * len(y))
            leny += len(y)
        return sum(losses) / leny

    def validate_loss(self):
        """
        Returns: weighted average loss over the validation set, or
        the data that is provided.
        
        """
        return self.loss_dl(self.valid_Xy)

    @property
    def eval_mode(self):
        """
        A ContextManager to put the model in evaluation mode
        """
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
        """
        A ContextManager to put the model in training mode
        """
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
        """
        Run the validation set (in evaluation mode) and store the loss and metrics into the evaluator.
        
        Arguments:
            pbar: tqdm progress bar (None)
                if not None, progress is reported on the progress bar
                
            log: dict
                additional labels to log when storing the results in the evaluator.
                
        Returns: float
            weighted average loss over the validation set
        """
        epochloss = 0
        n = 0
        epoch_y_pred = []
        epoch_y = []

        with self.eval_mode:
            for *X, y in self.valid_Xy:
                loss, y_pred = self._loss_xy(*X, y=y)
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
            
    def train_batch(self, *X, y=None):
        """
        Train the model on a single batch X, y. The model should already
        be in training mode.
        
        Arguments:
            *X: tensor
                inputs that are used by the forward of the model
            y: tensor
                ground truth labels, the predictions are compared against
        
        Returns: (float, tensor)
            a tuple with the loss for the predictions on X,
            and a tensor with the predicted values
        """
        self.optimizer.zero_grad()
        loss, y_pred = self._loss_xy(*X, y=y)
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
    
    def train(self, epochs, lr=None, report_frequency=None, save=None, optimizer=None, optimizer_params=None, scheduler=False, scheduler_params=None, weight_decay=None, momentum=None, save_lowest=None, save_highest=None, log={}):
        """
        Train the model for the given number of epochs. Loss and metrics
        are logged during training in an evaluator. If a model was already
        (partially) trained, training will continue where it was left off.
        
        Arguments:
            epochs: int
                the number of epochs to train the model
            
            lr: float, tuple of floats, or list of floats
                float: set the learning
                (upper, lower): switch the scheduler to OneCycleLR and
                    use a cyclic annealing learning rate
                    between an upper and lower bound.
                [upper, lower]: switch the scheduler to Linear Decay and
                    use a linearly decaying learning rate
                    between an upper and lower bound. 
            
            report_frequency: int
                configures after how many epochs the loss and metrics are
                logged and reported during training. This is remembered
                for consecutive calls to train.
            
            save: str (None)
                If not None, saves (commits) the model after each reported
                epoch, under the name 'save'-epochnr
            
            optimizer: PyTorch Optimizer (None)
                If not None, changes the optimizer class to use.

            optimizer_params: dict (None)
                If not None, the parameters to configure the optimizer.

            scheduler: None, custom scheduler class
                used to adapt the learning rate. Set OneCycleLR or Linear Decay
                through the learning rate. Otherwise, provide a custom
                class/function to initialize a scheduler by accepting
                (optimizer, learning_rate, scheduler_cycle)

            scheduler_params: dict (None)
                additional parameters that are passed when initializing the scheduler

            weight_decay: float
                Apply weight_decay regularization with the AdamW optimizer

            momentum: float
                Apply momentum with the AdamW optimizer

            save_lowest: bool (False)
                Automatically commit/save the model when reporting an epoch and the validation loss is lowest
                than seen before. The model is saved as 'lowest' and can be checked out by calling lowest()
                on the trainer.
        """
        
        self._scheduler_start = self.epochid # used by OneCycleScheduler
        self._scheduler_epochs = epochs
        self.scheduler_params = scheduler_params
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
                    try:
                        metric += f'{m}={value:.5f} '
                    except: pass
                print(f'{self.epochid:>{epochspaces}} {self._time():.2f}s trainloss={epochloss:.5f} validloss={validloss:.5f} {metric}')
                if save is not None:
                    self.commit(f'{save}-{self.epochid}')
                if save_lowest is not None:
                    if self.lowest_score is None or validloss < self.lowest_score:
                        self.lowest_score = validloss
                        self.commit('lowest')
    
    def lowest(self):
        """
        Checkout the model with the lowest validation loss, that was committed when training with save_lowest=True
        """
        self.checkout('lowest')

    def learning_curve(self, y='loss', series='phase', select=None, xlabel = None, ylabel = None, title=None, label_prefix='', **kwargs):
        """
        Plot a learning curve with the train and valid loss on the y-axis over the epoch on the x-axis. 
        The plot is generated by the evaluator that logged training progress. By default the evaluator logs:
        - epoch: the epoch number
        - phase: 'train' or 'valid'
        - loss: the weighted average loss
        under the name of each metric function, the resulting value when called with (y, y_pred)
        and the additional values that are passed to train() through the log parameter. 
        
        Arguments:
            y: str or function
                the metric that is used for the y-axis. It has to be a metric that was collected during training.
                if a function is passed, the name of the function is used.
            series: str ('phase')
                the label to use as a series. By default, 'phase' is used to plot both the train and valid results.
            select: see evaluator.select
                using the values 'train' and 'valid' you can select to plot only the train or valid sets.
            xlabel: str
                the label used on the x-axis
            ylabel: str
                the label used on the y-axis
            title: str
                the title of the plot
            label_prefix: str
                prefixes the label, so that you can combine a plot with results from different metrics or models
            **kwargs: dict
                forwarded to matplotlib's plot or scatter function
        """
        return self.evaluator.line_metric(x='epoch', series=series, select=select, y=y, xlabel = xlabel, ylabel = ylabel, title=title, label_prefix=label_prefix, **kwargs)
        
    def validation_curve(self, y=None, x='epoch', series='phase', select=None, xlabel = None, ylabel = None, title=None, label_prefix='', **kwargs):
        """
        Plot a metric for the train and valid set, over epoch on the x-axis. The plot is generated by the evaluator
        that logged training progress. By default the evaluator logs:
        - epoch: the epoch number
        - phase: 'train' or 'valid'
        - loss: the weighted average loss
        under the name of each metric function, the resulting value when called with (y, y_pred)
        and the additional values that are passed to train() through the log parameter. 
        
        Arguments:
            y: str or function
                the metric that is used for the y-axis. It has to be a metric that was collected during training.
                if a function is passed, the name of the function is used.
            x: str ('epoch')
                the label used for the x-axis.
            series: str ('phase')
                the label to use as a series. By default, 'phase' is used to plot both the train and valid results.
            select: see evaluator.select
                using the values 'train' and 'valid' you can select to plot only the train or valid sets.
            xlabel: str
                the label used on the x-axis
            ylabel: str
                the label used on the y-axis
            title: str
                the title of the plot
            label_prefix: str
                prefixes the label, so that you can combine a plot with results from different metrics or models
            **kwargs: dict
                forwarded to matplotlib's plot or scatter function
        """
        if y is not None and type(y) != str:
            y = y.__name__
        return self.evaluator.line_metric(x=x, series=series, select=select, y=y, xlabel = xlabel, ylabel = ylabel, title=title, label_prefix=label_prefix, **kwargs)
       
    def freeze(self, last=-1):
        """
        Mostly used for transfer learning, to freeze all parameters of a model, until the given layer (exclusive).
        
        Arguments:
            last: int (-1)
                Freeze all layers up to this layer number. -1 is the last layer.
        """
        for c in list(self.model.children())[:last]:
            for p in c.parameters():
                p.requires_grad=False

    def unfreeze(self):
        """
        Mostly used for transfer learning, to unfreeze all parameters of a model.
        """
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
        """
        Run a learning rate finder on the dataset (as propesed by Leslie Smith and implemented in FastAI). 
        This saves the model, then starting with a very low learning rate
        iteratively trains the model on a single mini-batch and logs the loss on the validation set. Gradually, the
        learning rate is raised. The idea is that the graph contains information on a stable setting of the learning
        rate. This does not always work, and often after some training, if learning is not stable, the learning rate
        still needs to be adjusted. 
        
        The result is a plot of the validation loss over the change in learning rate.
        
        Arguments:
            lr: [small float, big float] ([1e-6, 10])
                Interval of learning rates to inspect
            steps: int (40)
                number of (exponential) steps to divide the learning rate interval in
            smooth: float (0.05)
                smoothing parameter, to generate a more readable graph
            cache_valid: bool (True)
                whether to keep the validation set if possible in memory. Switch of if there is insufficient memory
        """
        with tuner(self, exprange(lr[0], lr[1], steps), self.set_lr, label='lr', yscale='log', smooth=smooth, cache_valid=cache_valid, **kwargs) as t:
            t.run()
