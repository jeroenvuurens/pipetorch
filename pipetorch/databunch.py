import numpy as np
import pandas as pd
import copy

class Databunch:
    def __init__(self, train_ds, valid_ds, test_ds=None, batch_size=32, num_workers=0, shuffle=True, pin_memory=False, scaler=None, balance=False):
        self.train_ds = train_ds
        self.valid_ds = valid_ds
        self.test_ds = test_ds
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.shuffle = shuffle
        self.pin_memory = pin_memory
        self.scaler = scaler
        self.balance = balance
        
    @property
    def train_dl(self):
        try:
            return self._train_dl
        except:
            from torch.utils.data import DataLoader

            sampler = self._weighted_sampler(self.balance) if self.balance is not False else None
            shuffle = False if self.balance is not False else self.shuffle
            self._train_dl = DataLoader(self.train_ds, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=shuffle, pin_memory=self.pin_memory, sampler=sampler)
            return self._train_dl

    @property
    def valid_dl(self):
        try:
            return self._valid_dl
        except:
            from torch.utils.data import DataLoader

            self._valid_dl = DataLoader(self.valid_ds, batch_size=len(self.valid_ds), num_workers=self.num_workers, shuffle=False, pin_memory=self.pin_memory)
            return self._valid_dl

    @property
    def test_dl(self):
        try:
            return self._test_dl
        except:
            from torch.utils.data import DataLoader

            self._test_dl = DataLoader(self.test_ds, batch_size=len(self.test_ds), num_workers=self.num_workers, shuffle=False, pin_memory=self.pin_memory)
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
        self._batch_size = min(value, len(self.train_ds))
        self.reset()

    @property
    def num_workers(self):
        return self._num_workers

    @num_workers.setter
    def num_workers(self, value):
        self._num_workers = value
        self.reset()
    
    def inverse_transform_y(self, y):
        return self.scaler.inverse_transform_y(y)

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
        return self.train_ds.y.tensors[1]

    @property
    def valid_y(self):
        return self.valid_ds.y.tensors[1]

    @property
    def test_y(self):
        return self.test_ds.y.tensors[1]
