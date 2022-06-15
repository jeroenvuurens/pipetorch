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
#from tqdm.notebook import tqdm
from ..evaluate.evaluate import Evaluator
from ..helper import run_magic
from torch.optim.lr_scheduler import OneCycleLR, ConstantLR
from .tuner import *
from .helper import nonondict, tqdm_trainer
from functools import partial
import os
    
def to_numpy(arr):
    try:
        return arr.data.cpu().numpy()
    except: pass
    try:
        return arr.to_numpy()
    except: pass
    return arr

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

POST_FORWARD = {nn.L1Loss:None, 
                nn.MSELoss:None, 
                nn.CrossEntropyLoss:lambda y: torch.argmax(y, dim=1),
                nn.NLLLoss: lambda y: torch.argmax(y, dim=1), 
                nn.PoissonNLLLoss: lambda y: torch.argmax(y, dim=1), 
                nn.GaussianNLLLoss: lambda y: torch.argmax(y, dim=1), 
                nn.KLDivLoss: lambda y: torch.argmax(y, dim=1), 
                nn.BCELoss: lambda y: torch.round(y), 
                nn.BCEWithLogitsLoss: lambda y: torch.round(y), 
                nn.HuberLoss: None,
                nn.SmoothL1Loss: None
               }
        
class Trainer:
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
            
        optimizer_params: dict (None)
            the parameters that are passed (along with the model parameters)
            to initialize an optimizer. A 'nonondict' is used, meaning that
            when a None value is set, the key is removed, so that the default
            value is used instead.
            
        scheduler: None, OneCycleLR, ConstantLR
            used to adapt the learning rate: 
            - None will use a constant learning rate
            - OneCycleLR will will use a cyclic annealing learning rate
              between an upper and lower bound.
            - ConstantLR will use a linear decaying learning rate between
              an upper bound and lower bound. You can optionally use
              'cycle' when calling 'train' to restart ConstantLR 
              every 'cycle' epochs.
              
        scheduler_params: dict (None)
            additional parameters that are passed when initializing the scheduler

        weight_decay: float
            Apply weight_decay regularization with the AdamW optimizer
            
        momentum: float
            Apply momentum with the AdamW optimizer
            
        random_state: int
            used to set a random state for reproducible results
            
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
            
        post_forward: func (None)
            For some projects, the loss function requires a different output than
            the metrics that are being used. 

            Example 1: For nn.BCELoss() the target value
            must be a likelihood, while accuracy_score requires a class label. 
            The model returns a likelihood with an nn.Sigmoid() on the ouput layer, 
            but the metrics can only be computed if the likelihood is converted into 
            a predicted label (e.g. torch.round() ). 

            Example 2: nn.CrossEntropyLoss() requires a distribution over the possible labels
            while multi-class evaluation matrics require the predicted class. This is commonly
            computed with torch.argmax(y, dim=1).

            To allow for this behavior, the trainer can use a post_forward fuction inbetween
            loss and metrics. It will attempt to use a post_forward in the following order: 
            - a function passed here
            - a post_forward method that is added to the model
            - infer a post_forward based on the loss function. 

            For inferring a post_forward based on
            the loss function, there is a dictionary in train.POST_FORWARD that covers the 
            most commonly used loss functions.

            If no post_forward is found, and the loss function is unknown, then None is used
            and a warning is printed. Pass post_forward=False to suppress this warning.
            
        debug: bool (False)
            stores X, y and y_pred in properties so that they can be inspected
            when an error is thrown.
    """
    def __init__(self, 
                 model, 
                 loss, 
                 *data, 
                 metrics = [], 
                 optimizer=AdamW, 
                 optimizer_params=None, 
                 scheduler=None, 
                 scheduler_params=None,
                 weight_decay=None, 
                 momentum=None, 
                 gpu=False,
                 random_state=None, 
                 evaluator=None, 
                 debug=False,
                 post_forward=None):
        
        # the amount of epochs in a cycle, 
        # validation is only done at the end of each cycle
        self.loss = loss
        self.random_state = random_state
        self.gpu(gpu)
        self.set_data(*data)
        self._model = model
        self._debug = debug
        self._set_post_forward(post_forward, model, loss)
        self.optimizer = optimizer
        self.optimizer_params = optimizer_params
        self.scheduler = scheduler
        self.scheduler_params = scheduler_params
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
            self.databunch = data[0]
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

    def _set_post_forward(self, post_forward, model, loss):
        if post_forward:
            self.post_forward = post_forward
            return
        if post_forward == False:
            self.post_forward = lambda y:y
        try:
            self.post_forward = model.post_forward
            return
        except:
            self.post_forward = lambda y:y
            for l, func in POST_FORWARD.items():
                try:
                    if loss.__class__ == l:
                        if func:
                            self.post_forward = func
                        return
                except: pass
        print('Warning, assuming no post_forward is needed (unknown loss function). Pass post_forward=False to suppress this warning.')
                    
    def reset_evaluator(self):
        try:
            del self._evaluator
        except: pass
        self.epochid = 0
                    
    @property
    def evaluator(self):
        """
        The (PipeTorch) evaluator that is used to log training progress
        """
        try:
            return self._evaluator
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
        self._gpu = self.device == 'cuda'

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
    def epochidstr(self):
        return f'{self.epochid:>{self._epochspaces}}' if self.cycle >= 1 else f'{self.subepochid:{self._epochspaces+3}.2f}'
        
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
            assert isinstance(value, dict), 'you have set optimizer_params to a dict'
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
            assert isinstance(value, dict), 'you have set scheduler_params to a dict'
            self._scheduler_params = nonondict(value)
        
    def del_optimizer(self):
        try:
            del self._optimizer
        except: pass
        self.del_scheduler()

    def del_scheduler(self):
        try:
            del self._scheduler
        except: pass

    def scheduler_step(self):
        try:
            self.scheduler.step()
        except ValueError:
            del self._scheduler
            self.scheduler.step()
        
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
                min_lr = self.min_lr
                max_lr = self.max_lr
                try:
                    min_lr = min_lr[0]  # working with parameter groups
                except: pass
                try:
                    max_lr = max_lr[0]  # working with parameter groups
                except: pass
                factor = (min_lr / max_lr) ** (1 / self._scheduler_epochs)
                self._scheduler = ConstantLR(self.optimizer, factor,
                                  self._scheduler_epochs, **self.scheduler_params)
            elif schedulerclass == OneCycleLR:
                scheduler_params = self.scheduler_params
                total_steps = math.ceil(len(self.train_dl) * self.cycle)
                self._scheduler = OneCycleLR(self.optimizer, 
                                  self.min_lr, total_steps=total_steps, **scheduler_params) 
            else:
                try:
                    self._scheduler = schedulerclass(self.optimizer, 
                                  self.lr, **self.scheduler_params)
                except:
                    raise NotImplementedError(f'The provided {schedulerclass} function does not work with ({self.optimizer}, {self.lr}, {self._scheduler_epochs}, {len(self.train_dl)}) to instantiate a scheduler')
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

    @model.setter
    def model(self, value):
        self._model = value
        try:
            del self._optimizer
        except: pass
        self.epochid = 0
    
    def parameters(self):
        """
        Prints the (trainable) model parameters
        """
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                print(name, param.data)

    def reset_model(self):
        """
        Resets all weights in the current model
        
        refs:
            - https://discuss.pytorch.org/t/how-to-re-set-alll-parameters-in-a-network/20819/6
            - https://stackoverflow.com/questions/63627997/reset-parameters-of-a-neural-network-in-pytorch
            - https://pytorch.org/docs/stable/generated/torch.nn.Module.html
        """

        @torch.no_grad()
        def weight_reset(m: nn.Module):
            # - check if the current module has reset_parameters & if it's callabed called it on m
            reset_parameters = getattr(m, "reset_parameters", None)
            if callable(reset_parameters):
                m.reset_parameters()

        # Applies fn recursively to every submodule see: https://pytorch.org/docs/stable/generated/torch.nn.Module.html
        self.model.apply(fn=weight_reset)
        self.epochid = 0
        try:
            del self._optimizer
        except: pass       
                
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
        if self._debug:
            self.lastx = X
            self.lastyfw = self.model(*X)
            return self.lastyfw
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
        self.post_forward(self.forward(*X))

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
        raise ValueError('The Trainer was somehow not initialized with a post_forward')

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
        self._commit[label] = (model_state, optimizer_state, self.subepochid, self.evaluator.results.clone())

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
        first have to initialize a model with the same configuration, and then use `Trainer.load(path)` to load
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
            model_state, optimizer_state, self.epochid, self.evaluator.results = self._commit.pop(label)
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
            model_state, optimizer_state, self.epochid, self.evaluator.results = self._commit[label]
            self.evaluator.results = self.evaluator.results[self.evaluator.results.epoch <= self.epochid]
            self.epochid = math.ceil(self.epochid)            
            self.model.load_state_dict(model_state)
            self.del_optimizer()            
            self.optimizer.load_state_dict(optimizer_state)  
        else:
            print('commit point {label} not found')

    def reset(self):
        """
        Resets the cached results, for tuning purposes.
        """
        self.reset_model()
        self.reset_evaluator()
            
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
            self._commit = { l:s for l,s in self._commit.items() if l == label }
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
        if self._debug:
            self.lasty = y
        y_pred = self.forward(*X)
        if self._debug:
            self.lastyfw = y_pred
        loss = self.loss(y_pred, y)
        return loss, y_pred 
    
    def _loss_forward_xy(self, *X, y=None):
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
        loss, y_pred = self._loss_xy(*X, y=y)
        y_pred = self.post_forward(y_pred)
        if self._debug:
            self.lastypfw
        return loss, y_pred 
    
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
            if self._debug:
                self.lasty = y
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
                loss, y_pred = self._loss_forward_xy(*X, y=y)
                epochloss += loss.item() * len(y_pred)
                n += len(y_pred)
                epoch_y_pred.append(to_numpy(y_pred))
                epoch_y.append(to_numpy(y))
                if pbar is not None:
                    pbar.update(self.valid_dl.batch_size)
            epochloss /= n
            epoch_y = np.concatenate(epoch_y, axis=0)
            epoch_y_pred = np.concatenate(epoch_y_pred, axis=0)
            metrics = self.evaluator._store_metrics(epoch_y, epoch_y_pred, 
                                                    annot={'phase':'valid', 'epoch':self.subepochid}, **log)
            self.evaluator._store_metric('loss', epochloss, 
                                         annot={'phase':'valid', 'epoch':self.subepochid}, **log)
        return epochloss, metrics

    def test(self):
        """
        Compute the metrics over the test set. 
        Requires a test_dl or databunch with a test_dl to be set on this Trainer.
        
        Returns: {}
            a dictionary with the computed metrics over the testset
        """
        epoch_y_pred = []
        epoch_y = []

        with self.eval_mode:
            for *X, y in self.test_Xy:
                loss, y_pred = self._loss_forward_xy(*X, y=y)
                epoch_y_pred.append(to_numpy(y_pred))
                epoch_y.append(to_numpy(y))
            epoch_y = np.concatenate(epoch_y, axis=0)
            epoch_y_pred = np.concatenate(epoch_y_pred, axis=0)
        return self.evaluator.compute_metrics(epoch_y, epoch_y_pred)
    
    def _test(self, pbar=None, log={}):
        """
        Run the test set (in evaluation mode) and store the loss and metrics into the evaluator.
        Is a helper function of train().
        
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
            for *X, y in self.test_Xy:
                loss, y_pred = self._loss_forward_xy(*X, y=y)
                epochloss += loss.item() * len(y_pred)
                n += len(y_pred)
                epoch_y_pred.append(to_numpy(y_pred))
                epoch_y.append(to_numpy(y))
                if pbar is not None:
                    pbar.update(self.test_dl.batch_size)
            epochloss /= n
            epoch_y = np.concatenate(epoch_y, axis=0)
            epoch_y_pred = np.concatenate(epoch_y_pred, axis=0)
            metrics = self.evaluator._store_metrics(epoch_y, epoch_y_pred, 
                                                    annot={'phase':'test', 'epoch':self.epochid}, **log)
            self.evaluator._store_metric('loss', epochloss, annot={'phase':'test', 'epoch':self.epochid}, **log)
        return epochloss, metrics
            
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
    
    def cross_validate(self, epochs, lr, cycle=1, silent=True, test=True, earlystop=False, reset_evaluator=True, log={}):
        """
        Only works with a Databunch from a DFrame that is configured for n-fold cross validation. 
        The model is trained n times (reinitializing every time), and the average metric is reported 
        over the trials.
        
        Arguments:
            epochs: int
                the maximum number of epochs to train. Training may be terminated early when
                convergence requirements are met.
            lr: float, (float, float) or [float, float]
                the learning rate to use for the optimzer. See lr for train().
            cycle: int (1)
                the number of epochs in a cycle. At the end of each cycle the validation is run.
            earlystop: int (False)
                terminates training when the validation loss has not improved for the last
                earlystop cycles.
            test: bool (False)
                run the test set every cycle (used for n-fold cross validation)
            log: {}
                see train(log), the cross validator extends the log with a folds column.
                
        """
        from ..data import Databunch
        
        data = self.databunch.iter_folds()
        folds = self.databunch.folds
        test_dl = self.test_dl if test else None
        pbar = tqdm_trainer(epochs, cycle, self.train_dl, self.valid_dl, test_dl, folds=folds)
        if reset_evaluator:
            self.reset_evaluator()
        
        
        def run(trainer, trial):
            trainer.reset_model()
            _ = next(data)
            log['fold'] = trial.number
            trainer.train(epochs, lr, cycle=cycle, pbar=pbar, log=log, 
                          test=test, silent=silent, earlystop=earlystop)
            return trainer.optimum(select=log)
        
        return self.optimize(run, n_trials=folds)
    
    def optimize(self, func, n_trials=None, timeout=None, catch=(), callbacks=None, 
                 gc_after_trial=False, show_progress_bar=False, grid=None):
        """
        Run n_trials on the given func to optimize settings and hyperparameters. This uses an 
        extension to the Optuna library tio create a study. This extension allows to define your
        trial func(trainer, trial) so that you can reuse the configured trainer. 
        
        Args:
            func: callable
                a function that is called to perform a trail and receives the Trainer and trial object.
                
            n_trials: int (None)
                number of trails to perform

            grid: dict (None)
                when grid is not None, a grid search is performed over the given values, e.g.
                grid={lr:[1e-3, 1e-2], batch-size=[32, 64]}

            For the other arguments, see Optuna.Study.optimize
        
        Returns: Study (extension to Optuna's Study)
            That contains the collected metrics for the trials
        """
        assert grid is None or n_trials is None, 'You cannot use Grid Search together with n_trials'
        study = self.study(grid=grid)
        study.optimize(func, n_trials=n_trials, timeout=timeout, catch=catch, callbacks=callbacks,
                      gc_after_trial=gc_after_trial, show_progress_bar=show_progress_bar)
        return study
    
    def train(self, epochs, lr=None, cycle=1, save=None, 
              optimizer=None, optimizer_params=None, scheduler=False, 
              scheduler_params=None, weight_decay=None, momentum=None, 
              save_lowest=False, save_highest=False, silent=False, pbar=None,
              targetloss=None, earlystop=False, log={}, test=False):
        """
        Train the model for the given number of epochs. Loss and metrics
        are logged during training in an evaluator. If a model was already
        (partially) trained, training will continue where it was left off.
        
        Args:
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
            
            cycle: int or float (1)
                Configures after how many epochs there are in a cycle. 
                the loss and metrics are logged and reported at the end of every cycle.
                For training on very large training sets, if cycle is set to a whole integer
                faction (e.g. cycle=1/10), then validation is done during after that part of
                every epoch. 
                The cycle setting is remembered for consecutive calls to train.
            
            silent: bool (False)
                whether to report progress. Note that even when silent=True
                the metrics are still logged at the end of every cycle.
            
            save: str (None)
                If not None, saves (commits) the model at the end of each cycle
                under the name 'save'-epochnr
            
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

            targetloss: float (None)
                terminates training when the validation loss drops below the targetloss.
                
            earlystop: int (False)
                terminates training when the validation loss has not improved for the last
                earlystop cycles.
                
            save_lowest: bool (False)
                when the validation loss is lower than seen before, the model is 
                saved/committed as 'lowest' and can be checked out by calling 
                lowest() on the trainer.
                
            log: {}
                At the end of every cycle, the loss and metrics over the train, valid
                and optionally test sets are computed and stored in a result set. The
                values passed in log are stored along with this metrics. Typically, this
                is used with a single trainer that is reused for several 'trials' 
                to analyze how the results changes. Several functions on the resultset
                allow to 'select' results based on these settings, are generate plots
                with these settings as 'series'.
                
            test: bool (False)
                run the test set every cycle (used for n-fold cross validation)
        """
        self.cycle = cycle
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
        self._cyclesnotimproved = 0
        self._lastvalidation = None
        model = self.model
        torch.set_grad_enabled(False)
        maxepoch = self.epochid + epochs
        self._epochspaces = int(math.log(maxepoch)/math.log(10)) + 1
        if pbar is None:
            if test:
                self.currentpbar = tqdm_trainer(epochs, cycle, self.train_dl, self.valid_dl, self.test_dl, silent=silent)
            else:
                self.currentpbar = tqdm_trainer(epochs, cycle, self.train_dl, self.valid_dl, silent=silent)
        else:
            self.currentpbar = pbar
        self._time()
        self._log_reset()
        check_scheduler = self.scheduler.__class__ == OneCycleLR
        if cycle < 1:
            log_batches = np.linspace(0, len(self.train_dl), int(round(1 / self.cycle)) + 1)[1:]
            log_batches = { int(round(b))-1 for b in log_batches }
        else:
            log_batches = { len(self.train_dl)-1 }
        log_next_epoch = self.epochid + cycle - 1 if cycle > 1 else self.epochid
        for i in range(epochs):
            with self.train_mode:
                for batch, (*X, y) in enumerate(self.train_Xy):
                    #print(self.cycle, len(self.train_dl), batch, log_batches, log_next_epoch, log_this_epoch)
                    if check_scheduler and self.scheduler._step_count == self.scheduler.total_steps:
                        self.del_scheduler()
                        self.scheduler
                    loss, y_pred = self.train_batch(*X, y=y)
                    self.scheduler.step()
                        
                    self.currentpbar.update(self.train_dl.batch_size)
                    #print(self.epochid, log_next_epoch, batch, log_batches)
                    if log_next_epoch == self.epochid:
                        y_pred = self.post_forward(y_pred)
                        if self._debug:
                            self.lastypfw = y_pred                          
                        self._log_increment(loss, y, y_pred)
                        if batch in log_batches:
                            if batch == len(self.train_dl) - 1:
                                batch = 0
                                self.epochid += 1
                                log_next_epoch = self.epochid + cycle - 1 if cycle > 1 else self.epochid
                            self.subepochid = self.epochid + batch / len(self.train_dl)
                            validloss, validmetrics = self.validate(pbar = self.currentpbar, log=log)
                            self._log(validloss, validmetrics, log, silent)
                            if save_lowest is not None and save_lowest:
                                if self.lowest_score is None or validloss < self.lowest_score:
                                    self.lowest_score = validloss
                                    self.commit('lowest')
                            if test:
                                self._test(pbar=self.currentpbar, log=log)

                            if save is not None and save:
                                self.commit(f'{save}-{self.epochid}')

                            if self._check_early_termination(validloss, targetloss, earlystop, silent):
                                break
                    elif batch == len(self.train_dl) - 1:
                        self.epochid += 1
        if pbar is None:
            try:
                self.currentpbar.close()    
            except: pass
    
    def _log_reset(self):
        self._epochloss = 0
        self._n = 0
        self._epoch_y_pred = []
        self._epoch_y = []

    def _log_increment(self, loss, y, y_pred):
        self._epochloss += loss.item() * len(y_pred)
        self._n += len(y_pred)
        self._epoch_y_pred.append(to_numpy(y_pred))
        self._epoch_y.append(to_numpy(y))
        
    def _log(self, validloss, validmetrics, log, silent):
        self._epochloss /= self._n
        self._epoch_y = np.concatenate(self._epoch_y, axis=0)
        self._epoch_y_pred = np.concatenate(self._epoch_y_pred, axis=0)
        metrics = self.evaluator._store_metrics(self._epoch_y, self._epoch_y_pred, 
                                                annot={'phase':'train', 'epoch':self.subepochid}, **log)
        self.evaluator._store_metric('loss', self._epochloss, annot={'phase':'train', 'epoch':self.subepochid}, **log)
        if not silent:
            reportmetric = ''
            for m in self.metrics:
                m = m.__name__
                value = validmetrics[m]
                try:
                    reportmetric += f'{m}={value:.5f} '
                except: pass
            print(f'{self.epochidstr} {self._time():.2f}s trainloss={self._epochloss:.5f} validloss={validloss:.5f} {reportmetric}')
        self._log_reset()

    def _check_early_termination(self, validloss, targetloss, earlystop, silent):
        if targetloss is not None and validloss <= targetloss:
            try:
                self.currentpbar.finish_fold()
            except: pass
            if not silent:
                print('Early terminating because the validation loss reached the target.')
            return True
        if earlystop:
            if self._lastvalidation is None:
                self._lastvalidation = validloss
            else:
                if validloss < self._lastvalidation:
                    self._cyclesnotimproved = 0
                else:
                    self._cyclesnotimproved += 1
                    if self._cyclesnotimproved >= earlystop:                                
                        try:
                            self.currentpbar.finish_fold()
                        except: pass
                        if not silent:
                            print(f'Early terminating because the validation loss has not improved the last {earlystop} cycles.')
                        return True
        
    def lowest(self):
        """
        Checkout the model with the lowest validation loss, that was committed when training with save_lowest=True
        """
        self.checkout('lowest')

    def debug(self):
        if self._debug:
            try:
                print('last X', self.lastx)
            except: pass
            try:
                print('last y', self.lasty)
            except: pass
            try:
                print('last model(X)', self.lastyfw)
            except: pass
            try:
                print('last post_forward(model(X))', self.lastypfw)
            except: pass
        
    def learning_curve(self, y='loss', series='phase', select=None, xlabel = None, ylabel = None, title=None, label_prefix='', fig=plt, legendargs={}, **kwargs):
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
            legendargs: dict ({})
                arguments that are passed to legend
            **kwargs: dict
                forwarded to matplotlib's plot or scatter function
        """
        return self.evaluator.line_metric(x='epoch', series=series, select=select, y=y, xlabel = xlabel, ylabel = ylabel, title=title, label_prefix=label_prefix, fig=fig, **kwargs)
        
    def validation_curve(self, y=None, x='epoch', series='phase', select=None, xlabel = None, ylabel = None, title=None, label_prefix='', fig=plt, **kwargs):
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
            fig: pyplot.Figure (None)
                the figure to put the plot in
            **kwargs: dict
                forwarded to matplotlib's plot or scatter function
        """
        if y is not None and type(y) != str:
            y = y.__name__
        return self.evaluator.line_metric(x=x, series=series, select=select, y=y, xlabel = xlabel, ylabel = ylabel, title=title, label_prefix=label_prefix, fig=fig, **kwargs)
       
    def curves(self, x='epoch', series='phase', select=None, xlabel = None, title=None, label_prefix='', **kwargs):
        m = len(self.metrics) + 1
        fig, ax = plt.subplots(nrows=1, ncols=m, figsize=(6 * m, 4))
        if title is not None:
            fig.title(title)
        for i, y in enumerate(['loss'] + self.metrics):
            self.validation_curve(y=y, x=x, series=series, select=select, xlabel=xlabel, fig=ax[i], **kwargs)
        
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

    def study(self, storage=None, sampler=None, pruner=None, study_name=None, direction=None, load_if_exists=False, directions=None, grid=None):
        """
        Creates an (extended) Optuna Study to study how hyperparameters affect the given target function 
        when training a model. This call will just instantiate and return the study object. Typical use is to
        first define a `trial` function, that will sample values to use as hyperparameters, instantiate and train a model,
        and return the optimal validation scores using `trainer.optimum`. Then call `study.optimize(trail, n_trials=)`
        to run the trial n_trial times. You can use `tuner.plot_hyperparameters()` to visualize the results, or any
        optuna method.
        
        If you want to create a study without optimizing for loss first, `Study.create_study` allows you to
        set the targets and directions.
        
        Arguments:
            grid: dict (None)
                dictionary with the values to use in a grid search
        
            for the arguments, see create_study in the Optuna library
            
        Returns:
            Study (which is a subclass of Optuna.study.Study)
        """
        from .study import Study
        return Study.create_study(self, storage=storage, sampler=sampler, pruner=pruner, study_name=study_name, direction=direction, load_if_exists=load_if_exists, directions=directions, grid=grid)
    
    def optimum(self, *target, direction=None, directions=None, **select):
        """
        Finds the cycle at which optimal results where obtained over the validation set, on the given optimization
        metric. 
        
        Args:
            *target: str or callable ('loss')
                names or metric functions that are used to decide what training cycle the model was most optimal

            direction: str or [ str ] (None)
                for every target: 'minimize' or 'maximize' to find the highest or lowest value on the given target
                If None, 'minimize' is used when optimize is 'loss', otherwise 'maximize' is used
 
            directions: [ str ] (None)
                same as direction, but now a list of 'minimize' or 'maximize' for multipe targets.
            
            select: {} (None)
                When None, the log={} values from the last call to train() are used. Otherwise, select
                is a dictionary with values that distinguish the results from the current trial 
                to the previous trails, which is needed to find the single best epoch of the current trail
                to return the metrics for that epoch.
                
        Returns:
            [ target ]
            A list of target values 
        """
        if len(target) == 0:
            target = ['loss'] + [ m.__name__ for m in self.metrics ]
        else:
            target = [ t.__name__ if callable(t) else t for t in target ]
            for t in target:
                try:
                    assert t == 'loss' or t in { m.__name__ for m in self.metrics }, \
                        f'Target {t} should be loss or a metric that is registered for the trainer'
                except:
                    assert False, f'Exception comparing target {t} to the registered metrics of the trainer'
        if direction is None and directions is None:
            if len(target) > 1:
                directions = [ 'minimize' if t == 'loss' else 'maximize' for t in target ]
            else:
                direction = 'minimize' if target[0] == 'loss' else 'maximize'
        r = self.evaluator.optimum(*target, direction=direction, directions=directions, **select)
        return [ r[t] for t in target ]
        
    def plot_hyperparameters(self, figsize=None):
        self.tuner.plot_hyperparameters(figsize)
        
    def tune_old(self, params,setter, lr=[1e-6, 1e-2], steps=40, smooth=0.05, label=None, **kwargs):
        lr_values = exprange(*lr, steps)
        if label is None:
            label = str(setter)
        if len(params) == 2:
            params = range3(*params)
        with tuner(self, lr_values, self.set_lr, smooth=0.05, label=label) as t:
            t.run_multi(params, setter)

    def tune_weight_decay_old(self, lr=[1e-6,1e-4], params=[1e-6, 1], steps=40, smooth=0.05, yscale='log', **kwargs):
        self.tune( params, partial(self.set_optimizer_param, 'weight_decay'), lr=lr, steps=steps, smooth=smooth, label='weight decay', yscale=yscale, **kwargs)

    def lr_find(self, lr=[1e-6, 10], steps=40, smooth=0.05, cache_valid=True, interactive=False, **kwargs):
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
                
            interactive: bool (False)
                switches the backend to matplotlib notebook to show the plot during training and switches
                to matplotlib inline when it is done. It cannot (yet) detect the previous backend, so this
                will only work when inline is the defaut mode.
        """
        if interactive:
            run_magic('matplotlib', 'notebook')
        with tuner(self, exprange(lr[0], lr[1], steps), self.set_lr, label='lr', yscale='log', smooth=smooth, cache_valid=cache_valid, **kwargs) as t:
            t.run()
        if interactive:
            run_magic('matplotlib', 'inline')
