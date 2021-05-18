import pandas as pd
import numpy as np
import seaborn as sns
import torch
import pickle
import matplotlib.pyplot as plt
from ..kernel.helper import *
from pandas import DataFrame
from pandas.core.groupby.generic import DataFrameGroupBy
from torch.utils.data import TensorDataset, Dataset, DataLoader, SubsetRandomSampler, Sampler, ConcatDataset
from torch.utils.data._utils.collate import default_collate
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
identity=lambda x:x

def map_categories(df, column):
    df = df[column]
    try:
        s = sorted(df.unique())
    except:
        s = df.unique()
    d = { v:i for i, v in enumerate(s) }
    df = df.map(d).astype(np.long)
    return df

class Databunch:
    def __init__(self, train_ds, valid_ds, test_ds=None, batch_size=32, num_workers=0, shuffle=True, pin_memory=False):
        self.train_ds = train_ds
        self.valid_ds = valid_ds
        self.test_ds = test_ds
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.shuffle = shuffle
        self.pin_memory = pin_memory

    @property
    def train_dl(self):
        try:
            return self._train_dl
        except:
            self._train_dl = DataLoader(self.train_ds, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=self.shuffle, pin_memory=self.pin_memory)
            return self._train_dl

    @property
    def valid_dl(self):
        try:
            return self._valid_dl
        except:
            self._valid_dl = DataLoader(self.valid_ds, batch_size=len(self.valid_ds), num_workers=self.num_workers, shuffle=False, pin_memory=self.pin_memory)
            return self._valid_dl

    @property
    def test_dl(self):
        try:
            return self._test_dl
        except:
            self._test_dl = DataLoader(self.test_ds, batch_size=len(self.test_ds), num_workers=self.num_workers, shuffle=False, pin_memory=self.pin_memory)
            return self._test_dl

    def inverse_transform_y(self, y):
        return self.train_ds.inverse_transform_y(y)

    @classmethod
    def from_df( cls, df, target, *features, split=0.2, window=None, shift=1, scale=True, separate_y=False, batch_size=64, **kwargs ):
        features = expand_features(df, target, *features)
        d = pd.concat([df[features], df[[target]]], axis=1)
        test_ds = None
        dss = SequenceDataset.create(d, split=split, window=window, shift=shift, scale=scale, separate_y=separate_y)
        if len(dss) == 2:
            train_ds, valid_ds = dss
        else:
            train_ds, valid_ds, test_ds = dss
        return cls(train_ds=train_ds, valid_ds=valid_ds, test_ds=test_ds, batch_size=batch_size, **kwargs)

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

    def sample(self, device=None):
        arrays = next(iter(self.train_dl))
        if device is not None:
            arrays = [ a.to(device) for a in arrays ]
        return arrays
    
    @property
    def train_X(self):
        return self.train_ds.X

    @property
    def valid_X(self):
        return self.valid_ds.X

    @property
    def test_X(self):
        return self.test_ds.X

    @property
    def train_y(self):
        return self.train_ds.y

    @property
    def valid_y(self):
        return self.valid_ds.y
    
    @property
    def test_y(self):
        return self.test_ds.y

    def _plot(self, **kwargs):
        figure = plt.figure(**kwargs)
        return plt.axes()

