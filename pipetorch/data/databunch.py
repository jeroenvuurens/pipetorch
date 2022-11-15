import numpy as np
import pandas as pd
import copy
from torch.utils.data import DataLoader, Subset, TensorDataset, ConcatDataset
from torchvision.datasets import VisionDataset
from .transformabledataset import TransformableDataset
from ..evaluate.evaluate import Evaluator

def traverse_dataset(ds):
    if isinstance(ds, Subset):
        return traverse_dataset(ds.dataset)
    if isinstance(ds, TransformableDataset):
        return traverse_dataset(ds.dataset)
    if isinstance(ds, TensorDataset):
        return ds.tensors
    if isinstance(ds, VisionDataset):
        try:
            return ds.data, ds.targets
        except:
            raise ValueError('Cannot traverse a VisionDataset')
    raise ValueError('Cannot traverse this type of Dataset')

class Databunch:
    """
    Following the idea of databunches from the Fast.ai library, 
    a Databunch is a convenient wrapper for a train and valid DataLoader in one object.
    This makes it less redundant to configure the dataloaders and ensures that the dataloaders for
    DFrame are always paired.
    
    Args:
        df: DFrame
            the source
        
        train_ds: DataSet
            A PyTorch DataSet for the train part
            
        valid_ds: DataSet
            A PyTorch DataSet for the valid part
            
        test_ds: DataSet
            A PyTorch DataSet for the test part
            
        batch_size: int (32)
            the batch size that is used to generate a PyTorch DataLoader for the train set
            
        valid_batch_size: int (None)
            the batch size that is used to generate a PyTorch DataLoader for the valid set.
            When None, batch_size is used.
            
        num_workers: int (2)
            the number of CPU cores that are assigned by the DataLoader to prepare data
            
        
            
    """
    
    def __init__(self, df, train_ds, valid_ds=None, test_ds=None, batch_size=32, valid_batch_size=None, 
                 num_workers=2, shuffle=True, pin_memory=False, balance=False, collate=None):
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

    @property
    def folds(self):
        return self.df._pt_folds
        
    def iter_folds(self):
        if self.folds is None:
            yield self.train_dl, self.valid_dl, self.test_dl
        else:
            for fold in range(self.folds):
                yield self.fold(fold)

    def fold(self, fold):
        db = self.df.fold(fold).to_databunch(batch_size=self.batch_size, num_workers=self.num_workers, 
                                           shuffle=self.shuffle, pin_memory=self.pin_memory, balance=self.balance)
        return db.train_dl, db.valid_dl, db.test_dl
                
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
    
    def inverse_scale_y(self, y):
        return self.df.inverse_scale_y(y)
    
    def inverse_scale(self, X, y, y_pred, cum=None):
        return self.df.inverse_scale(X, y, y_pred, cum=cum)
    
    def from_numpy(self, X):
        X = self.df.from_numpy(X)
    
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
            df = self.inverse_scale(*X, y, y_pred, df)
        torch.set_grad_enabled(prev)
        return df
    
    def predict_train(self, model, device=None):
        return self.predict(model, self.train_dl, device=device)
    
    def predict_valid(self, model, device=None):
        return self.predict(model, self.valid_dl, device=device)

    def predict_test(self, model, device=None):
        return self.predict(model, self.test_dl, device=device)

    def sample(self, device=None, valid=False, test=False):
        """
        returns a single batch from the DataLoader.
        
        Args:
            device: torch.device (None)
                transfers the data to the given device, e.g. db.sample(torch.device('cuda:0'))
            
            valid: bool (False)
                retrieves a sample from the validation set
                
            test: bool (False)
                retrieves a sample from the test set
                
        Returns: (tensor, tensor)
            containing the input and output features for a retrieved batch from the dataset
        """
        arrays = next(iter(self.train_dl))
        if device is not None:
            arrays = [ a.to(device) for a in arrays ]
        return arrays

    @property
    def train_X(self):
        return traverse_dataset(self.train_ds)[0]

    @property
    def valid_X(self):
        return traverse_dataset(self.valid_ds)[0]

    @property
    def test_X(self):
        return traverse_dataset(self.test_ds)[0]

    @property
    def train_y(self):
        return traverse_dataset(self.train_ds)[-1]

    @property
    def valid_y(self):
        return traverse_dataset(self.valid_ds)[-1]

    @property
    def test_y(self):
        return traverse_dataset(self.test_ds)[-1]
    
    def to_evaluator(self, *metrics):
        return Evaluator(self.df, *metrics)
    
    def df_to_dset(self, df):
        """
        Converts a DataFrame to a DSet that has the pipeline as this DataBunch.
        
        Arguments:
            df: DataFrame
        
        Returns: DSet
        """
        return self.df.df_to_dset(df)

    def df_to_dataset(self, df):
        """
        Converts the given df to a DataSet using the pipeline of this DataBunch.
        
        Arguments:
            df: DataFrame or DFrame
                to convert into a DataSet
        
        returns: DataSet.
        """
        return self.df.df_to_dset(df).to_dataset()
    
    def df_to_dataloader(self, df):
        """
        Converts the given df to a DataLoader using the pipeline of this DataBunch.
        
        Arguments:
            df: DataFrame or DFrame
                to convert into a DataSet
        
        returns: DataSet.
        """
        return self._dataloader(self.df_to_dataset(df))    
