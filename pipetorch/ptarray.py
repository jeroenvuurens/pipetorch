import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
import copy
from .databunch import *
from .evaluate import *

def to_numpy(X):
    try:
        return arr.data.cpu().numpy()
    except: pass
    return arr

class ptarray(np.ndarray):
    _metadata = ['_pt_scaler', '_pt_indices', '_train_indices', '_valid_indices', '_test_indices', '_ycolumn', '_columns', '_bias']

    def __new__(cls, input_array):
        return np.asarray(input_array).view(cls)

    def __array_finalize__(self, obj) -> None:
        if obj is None: return
        d = { a:getattr(obj, a) for a in self._metadata if hasattr(obj, a) }
        self.__dict__.update(d)

    def __array_function__(self, func, types, *args, **kwargs):
        return self._wrap(super().__array_function__(func, types, *args, **kwargs))
        
    def __getitem__(self, item):
        r = super().__getitem__(item)
        if type(item) == tuple and len(item) == 2 and type(item[0]) == slice:
            r._columns = r._columns[item[1]]
            if self._check_list_attr('_pt_scaler'):
                r._pt_scaler = r._pt_scaler[item[1]]
        return r
        
    def __array_ufunc__(self, ufunc, method, *inputs, **kwargs):
        def cast(i):
            if type(i) is PTArray:
                return i.view(np.ndarray)
            return i
        
        inputs = [ cast(i) for i in inputs ]
        return self._wrap(super().__array_ufunc__(ufunc, method, *inputs, **kwargs))        
    
    def _check_list_attr(self, attr):
        try:
            return hasattr(self, attr) and len(getattr(self, attr)) > 0
        except:
            return False
    
    def _pt_scaler_exists(self):
        return self._check_list_attr('_pt_scaler')

    def _test_indices_exists(self):
        return self._check_list_attr('_test_indices')

#     def add_bias(self):
#         assert not hasattr(self, '_bias'), 'You cannot add a bias twice'
#         self._bias = 1
#         r = self._wrap(np.concatenate([np.ones((self.shape[0], 1)), self], axis=1))
#         r._bias = 1
#         r._columns = ['bias'] + r._columns
#         if r._pt_scaler_exists():
#             r._pt_scaler = [None] + r._pt_scaler
#         return r
    
    def ycolumn(self, columns):
        r = copy.copy(self)
        r._ycolumn = columns
        return r
    
    @property
    def yscaler(self):
        return [ s for s, c in zip(self._pt_scaler, self._columns) if c in self._ycolumn ]
    
    @property
    def _ycolumnsnr(self):
        return [ i for i, c in enumerate(self._columns) if c in self._ycolumn ]
    
    @property
    def _xcolumns(self):
        return [ c for c in self._columns if c not in self._ycolumn ]
    
    @property
    def _xcolumnsnr(self):
        return [ i for i, c in enumerate(self._columns) if c not in self._ycolumn ]

    def _train_indices_exists(self):
        return self._check_list_attr('_train_indices')
    
    def _valid_indices_exists(self):
        return self._check_list_attr('_valid_indices')
    
    def _wrap(self, a):
        a = PTArray(a)
        a.__dict__.update(self.__dict__)
        return a

    def polynomials(self, degree):
        assert not self._pt_scaler_exists(), "Run polynomials before scaling"
        poly = PolynomialFeatures(degree, include_bias=False)
        p = poly.fit_transform(self[:,:-self.ycolumns])
        return self._wrap(np.concatenate([p, self[:, -self.ycolumns:]], axis=1))
    
    def to_arrays(self):
        if self._test_indices_exists():
            return self.train_X, self.valid_X, self.test_X, self.train_y, self.valid_y, self.test_y
        elif self._valid_indices_exists():
            return self.train_X, self.valid_X, self.train_y, self.valid_y
        else:
            return self.train_X, self.train_y

    def scale(self, scalertype=StandardScaler):
        assert self._train_indices_exists(), "Split the DataFrame before scaling!"
        assert not self._pt_scaler_exists(), "Trying to scale twice, which is a really bad idea!"
        r = self._wrap(copy.deepcopy(self))
        r._pt_scaler = tuple(self._create_scaler(scalertype, column) for column in self[self._train_indices].T)
        return r.transform(self)

    @staticmethod
    def _create_scaler(scalertype, column):
        scaler = scalertype()
        scaler.fit(column.reshape(-1,1))
        return scaler

    def transform(self, array):
        out = []
        for column, scaler in zip(array.T, self._pt_scaler):
            if scaler is not None:
                out.append(scaler.transform(column.reshape(-1,1)))
            else:
                out.append(column)
        return self._wrap(np.concatenate(out, axis=1))

    def inverse_transform_y(self, y):
        y = to_numpy(y)
        y = y.reshape(-1, len(self._ycolumns))
        out = [ y[i] if self._pt_scaler[-self._ycolumns+i] is None else self._pt_scaler[-self._ycolumns+i].inverse_transform(y[:,i]) for i in range(y.shape[1]) ]
        if len(out) == 1:
            return self._wrap(out[0])
        return self._wrap(np.concatenate(out, axis=1))
    
    def inverse_transform_X(self, X):
        X = to_numpy(X)
        transform = [ X[i] if self._pt_scaler[i] is None else self._pt_scaler[i].inverse_transform(X[:,i]) for i in range(X.shape[1]) ]
        return np._wrap(np.concatenate(transform, axis=1))

    def inverse_transform(self, X, y):
        y = PTDataFrame(self.inverse_transform_y(y), columns=self._ycolumns)
        X = PTDataFrame(self.inverse_transform_X(X), columns=self._xcolumns)
        return pd.concat([X, y], axis=1)  

    def to_evaluator(self, metrics=[]):
        return Evaluator(*self.to_arrays(), metrics=metrics)
    
    def to_dataset(self):
        """
        returns: a list with a train, valid and test DataSet. Every DataSet contains an X and y, where the 
        input data matrix X contains all columns but the last, and the target y contains the last column
        columns: list of columns to convert, the last column is always the target. default=None means all columns.
        """
        import torch
        from torch.utils.data import TensorDataset, DataLoader
        tensor_y = torch.from_numpy(self.y)
        tensor_X = torch.from_numpy(self.X)

        p = [ TensorDataset(tensor_X[self._train_indices], tensor_y[self._train_indices]) ]
        if self._valid_indices_exists():
            p.append( TensorDataset(tensor_X[self._valid_indices], tensor_y[self._valid_indices]) )
        if self._test_indices_exists():
            p.append( TensorDataset(tensor_X[self._test_indices], tensor_y[self._test_indices]) )
        return p
    
    def to_databunch(self, batch_size=32, num_workers=0, shuffle=True, pin_memory=False, balance=False):
        """
        returns: a Databunch that contains dataloaders for the train, valid and test part.
        batch_size, num_workers, shuffle, pin_memory: see Databunch/Dataloader constructor
        """
        ds = self.to_dataset()
        return Databunch(*ds, batch_size=batch_size, num_workers=num_workers, shuffle=shuffle, pin_memory=pin_memory, scaler=self, balance=balance)    
   
    @property
    def X(self):
        return self[self._pt_indices][:,self._xcolumnsnr]    
    
    @property
    def y(self):
        return self[self._pt_indices][:,self._ycolumnsnr]    
    
    @property
    def train_X(self):
        return self[self._train_indices][:,self._xcolumnsnr]
    
    @property
    def valid_X(self):
        return self[self._valid_indices][:,self._xcolumnsnr]

    @property
    def test_X(self):
        return self[self._test_indices][:,self._xcolumnsnr]
    
    @property
    def train_y(self):
        return self[self._train_indices][:,self._ycolumnsnr]
    
    @property
    def valid_y(self):
        return self[self._valid_indices][:,self._ycolumnsnr]

    @property
    def test_y(self):
        return self[self._test_indices][:,self._ycolumnsnr]
        
