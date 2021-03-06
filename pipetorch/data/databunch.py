import numpy as np
import pandas as pd
import copy
from torch.utils.data import DataLoader
from ..evaluate.evaluate import Evaluator

class Databunch:
    def __init__(self, df, train_ds, valid_ds=None, test_ds=None, batch_size=32, valid_batch_size=None, num_workers=0, shuffle=True, pin_memory=False, balance=False, collate=None):
        self.df = df
        self.train_ds = train_ds
        self.valid_ds = valid_ds
        self.test_ds = test_ds
        self.batch_size = batch_size
        self.valid_batch_size = valid_batch_size
        self.num_workers = num_workers
        self.shuffle = shuffle
        self.pin_memory = pin_memory
        self.balance = balance
        self.collate = collate
        
    def _dataloader(self, ds, batch_size=32, shuffle=True, **kwargs):
        if self.collate is not None:
            kwargs['collate_fn'] = self.collate
        return DataLoader(ds, batch_size=batch_size, num_workers=self.num_workers, shuffle=shuffle, pin_memory=self.pin_memory, **kwargs)
        
    @property
    def train_dl(self):
        try:
            return self._train_dl
        except:
            kwargs = {}
            if self.balance:
                kwargs['sampler'] = self._weighted_sampler(self.balance)
            shuffle = False if self.balance is not False else self.shuffle
            self._train_dl = self._dataloader(self.train_ds, batch_size=self.batch_size, shuffle=shuffle, **kwargs)
            return self._train_dl

    @property
    def valid_dl(self):
        try:
            return self._valid_dl
        except:
            self._valid_dl = self._dataloader(self.valid_ds, batch_size=self.valid_batch_size, shuffle=False)
            return self._valid_dl

    @property
    def test_dl(self):
        try:
            return self._test_dl
        except:
            self._test_dl = self._dataloader(self.test_ds, batch_size=self.valid_batch_size, shuffle=False)
            return self._test_dl

    def reset(self):
        try:
            del self._valid_dl
        except: pass
        try:
            del self._train_dl
        except: pass
        try:
            del self._test_dl
        except: pass

    def _weighted_sampler(self, weights):
        import torch
        from torch.utils.data import WeightedRandomSampler

        target = self.train_ds.tensors[1].numpy().squeeze()
        if weights == True:
            weights = {t:(1. / c) for t, c in zip(*np.unique(target, return_counts=True))}
        samples_weight = np.array([weights[t] for t in target])
        samples_weight = torch.from_numpy(samples_weight)
        return WeightedRandomSampler(samples_weight, len(samples_weight))
      
    @property
    def batch_size(self):
        return self._batch_size
        
    @batch_size.setter
    def batch_size(self, value):
        if value is not None:
            self._batch_size = min(value, len(self.train_ds))
            self.reset()

    @property
    def valid_batch_size(self):
        try:
            return min(self._valid_batch_size, len(self.valid_ds))
        except:
            return len(self.valid_ds)
        
    @property
    def test_batch_size(self):
        try:
            return min(self._test_batch_size, len(self.test_ds))
        except:
            return len(self.test_ds)
        
    @valid_batch_size.setter
    def valid_batch_size(self, value):
        if value is not None:
            self._valid_batch_size = value
            self._test_batch_size = value
            self.reset()

    @property
    def shuffle(self):
        return self._shuffle
        
    @shuffle.setter
    def shuffle(self, value):
        if value is not None:
            self._shuffle = value

    @property
    def collate(self):
        try:
            return self._collate
        except: pass

    @collate.setter
    def collate(self, value):
        if value is not None:
            self._collate = value

    @property
    def balance(self):
        return self._balance

    @balance.setter
    def balance(self, value):
        self._balance = value
        self.reset()
            
    @property
    def num_workers(self):
        return self._num_workers

    @num_workers.setter
    def num_workers(self, value):
        if value is not None:
            self._num_workers = value
            self.reset()
    
    def inverse_transform_y(self, y):
        return self.df.inverse_transform_y(y)
    
    def inverse_transform(self, X, y, y_pred, cum=None):
        return self.df.inverse_transform(X, y, y_pred, cum=cum)
    
    def predict(self, model, dl, device=None):
        import torch
        model.eval()
        prev = torch.is_grad_enabled()
        torch.set_grad_enabled(False)
        if device is None:
            try:
                device = model.device
            except: pass
        df = None
        for *X, y in dl:
            if device is not None:
                X = [ x.to(device) for x in X ]
            y_pred = model(*X)
            df = self.inverse_transform(*X, y, y_pred, df)
        torch.set_grad_enabled(prev)
        return df
    
    def predict_train(self, model, device=None):
        return self.predict(model, self.train_dl, device=device)
    
    def predict_valid(self, model, device=None):
        return self.predict(model, self.valid_dl, device=device)

    def predict_test(self, model, device=None):
        return self.predict(model, self.test_dl, device=device)

    def sample(self, device=None):
        arrays = next(iter(self.train_dl))
        if device is not None:
            arrays = [ a.to(device) for a in arrays ]
        return arrays

    @property
    def train_X(self):
        return self.train_ds.tensors[0]

    @property
    def valid_X(self):
        return self.valid_ds.tensors[0]

    @property
    def test_X(self):
        return self.test_ds.tensors[0]

    @property
    def train_y(self):
        return self.train_ds.tensors[1]

    @property
    def valid_y(self):
        return self.valid_ds.tensors[1]

    @property
    def test_y(self):
        return self.test_ds.tensors[1]
    
    def to_evaluator(self, *metrics):
        return Evaluator(self.df, *metrics)