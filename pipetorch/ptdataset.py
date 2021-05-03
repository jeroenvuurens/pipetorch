import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
import matplotlib.pyplot as plt
import copy
import path
from pathlib import Path
import os

class PTDataSet(pd.DataFrame):
    _metadata = ['_df', '_dfindices', '_pt_scalerx', '_pt_scalery', '_pt_categoryx', '_pt_categoryy', '_pt_columny', '_pt_columnx', '_pt_transposey', '_pt_bias', '_pt_polynomials', '_pt_evaluator', '_dtype', '_pt_sequence_window', '_pt_sequence_shift_y']

    _internal_names = pd.DataFrame._internal_names + ['_X__tensor', '_y__tensor', '_tensor__indices']
    _internal_names_set = set(_internal_names)

    @property
    def _constructor(self):
        return PTDataSet
       
    @classmethod
    def from_ptdataframe(cls, data, df, dfindices):
        r = cls(data)
        r._df = df
        r._dfindices = dfindices
        r._pt_scalerx = df._scalerx
        r._pt_scalery = df._scalery
        r._pt_categoryx = df._categoryx
        r._pt_categoryy = df._categoryy
        r._pt_columny = df._columny
        r._pt_columnx = df._columnx
        r._pt_transposey = df._transposey
        r._pt_polynomials = df._pt_polynomials
        r._pt_bias = df._pt_bias
        r._dtype = df._dtype
        r._pt_sequence_window = df._pt_sequence_window
        r._pt_sequence_shift_y = df._pt_sequence_shift_y
        return r
    
    def to_ptdataframe(self):
        cls = self._df.__class__
        r = cls(self)
        for c in r._metadata:
            r.__setattr__(c, self._df.__getattr__(c))
        for c in ['_pt__scale_columns', '_pt__scalerx', '_pt__scalery', '_pt__categoryx', '_pt__categoryy']:
            try:
                r.__setattr__(c, self._df.__getattr__(c))
            except: pass
        r._pt__train = self
        r._pt__full = self
        r._pt__valid = None
        r._pt__test = None
        r._pt__indices = list(range(len(self)))
        r._pt__train_indices = r._pt__indices
        r._pt__valid_indices = []
        r._pt__test_indices = []
        r._pt_split = None
        r._pt_random_state = None
        r._pt_balance = None
        r._pt_shuffle = False
        r._pt_valid_dataframe = self[:0]
        r._pt_test_dataframe = self[:0]
        return r

    @property
    def _scalerx(self):
        return self._pt_scalerx
        
    @property
    def _scalery(self):
        return self._pt_scalery

    @property
    def _categoryx(self):
        return self._pt_categoryx
        
    @property
    def _categoryy(self):
        return self._pt_categoryy

    @property
    def _shift_y(self):
        if self._pt_sequence_shift_y is not None:
            return self._pt_sequence_shift_y
        else:
            return 0
    
    @property
    def _sequence_window(self):
        try:
            if self._pt_sequence_window is not None:
                return self._pt_sequence_window
        except:pass
        return 1
    
    @property
    def _sequence_index_y(self):
        return self._pt_sequence_window+self._shift_y-1

    @property
    def _columny(self):
        return [ self.columns[-1] ] if self._pt_columny is None else self._pt_columny
        
    @property
    def _transposey(self):
        return True if self._pt_transposey is None else self._pt_transposey
            
    @property
    def _columnx(self):
        if self._pt_columnx is None:
            return [ c for c in self.columns if c not in self._columny ]
        return self._pt_columnx
   
    @property
    def _polynomials(self):
        return self._pt_polynomials
       
    @property
    def _bias(self):
        return self._pt_bias
    
    def _transform(self, scalers, array):
        out = []
        for i, scaler in enumerate(scalers):
            if scaler is not None:
                out.append(scaler.transform(array[:, i:i+1]))
            else:
                out.append(array[:, i:i+1])
        return np.concatenate(out, axis=1)

    def resample(self, n=True):
        if n == True:
            n = len(self)
        if n < 1:
            n = n * len(self)
        return self[resample(list(range(len(self))), n_samples = int(n))]
    
    def interpolate(self, factor=2, sortcolumn=None):
        if not sortcolumn:
            sortcolumn = self.columns[0]
        df = self.sort_values(by=sortcolumn)
        for i in range(factor):
            i = df.rolling(2).sum()[1:] / 2.0
            df = pd.concat([df, i], axis=0)
            df = df.sort_values(by=sortcolumn)
        return self._df._ptdataset(df)
    
    @property
    def _x_category(self):
        if self._pt_sequence_window is not None:
            self = self.iloc[:-self._shift_y]
        if self._categoryx is None:
            return self[self._columnx]
        r = copy.copy(self[self._columnx])
        for c, cat in zip(r._columnx, r._categoryx):
            if cat is not None:
                r[c] = cat.transform(r[c])
        return r
    
    @property
    def _x_numpy(self):
        return self._x_category.to_numpy()
    
    @property
    def _x_polynomials(self):
        try:
            return self._polynomials.fit_transform(self._x_numpy)
        except:
            return self._x_numpy

    @property
    def _x_scaled(self):
        if len(self) > 0:
            return self._transform(self._scalerx(), self._x_polynomials)
        return self._x_polynomials
            
    @property
    def _x_biased(self):
        a = self._x_scaled
        if self._bias:
            return np.concatenate([np.ones((len(a),1)), a], axis=1)
        return a
    
    @property
    def X(self):
        return self._x_biased

    @property
    def X_tensor(self):
        return self.tensors[0]

    @property
    def y_tensor(self):
        return self.tensors[1]
 
    @property
    def tensors(self):
        try:
            return self._X__tensor, self._y__tensor
        except:
            X = self.X.astype(np.float32) if self._dtype is None else self.X
            y = self.y.astype(np.float32) if self._dtype is None else self.y
            if self._pt_sequence_window is None:
                from .ptsequencetensor import tensors
                self._X__tensor, self._y__tensor, self._tensor__indices = tensors(X, y)
            else:
                from .ptsequencetensor import sequence_tensors
                self._X__tensor, self._y__tensor, self._tensor__indices = sequence_tensors(X, y, self._sequence_window)
            return self._X__tensor, self._y__tensor
            
    @property
    def _range_y(self):
        stop = len(self) if self._shift_y >= 0 else len(self) + self._shift_y
        start = min(stop, self._sequence_window + self._shift_y - 1)
        return slice(start, stop)
        
    @property
    def _y_category(self):
        if self._pt_sequence_window is not None:
            self = self.iloc[self._range_y]
        if self._categoryy is None:
            return self[self._columny]
        r = copy.copy(self[self._columny])
        for c, cat in zip(r._columny, r._categoryy):
            if cat is not None:
                r[c] = cat.transform(r[c])
        return r
    
    @property
    def _y_numpy(self):
        return self._y_category.to_numpy()
    
    @property
    def _y_scaled(self):
        if len(self) > 0:
            return self._transform(self._scalery(), self._y_numpy)
        return self._y_numpy
            
    @property
    def _y_transposed(self):
        return self._y_scaled.squeeze() if self._transposey else self._y_scaled

    @property
    def y(self):
        return self._y_transposed
    
    def to_dataset(self):
        """
        returns: a list with a train, valid and test DataSet. Every DataSet contains an X and y, where the 
        input data matrix X contains all columns but the last, and the target y contains the last column
        columns: list of columns to convert, the last column is always the target. default=None means all columns.
        """
        from torch.utils.data import TensorDataset
        return TensorDataset(*self.tensors)
    
    def predict(self, predict):
        y_pred = predict(self.X)
        return self._df.inverse_transform(self.X, self.y, y_pred)
    
    def add_column(self, y_pred, erase_y=True, columns=None):
        if self._tensor__indices is not None:
            offset = self._range_y.start
            indices = [ i + offset for i in self._tensor__indices ]
        else:
            indices = list(range(len(y_pred)))
        assert len(y_pred) == len(indices), f'The number of predictions ({len(y_pred)}) does not match the number of samples ({len(indices)})'
        df = self.to_ptdataframe()
        return df.add_column(y_pred, indices, erase_y=erase_y, columns=columns)
    
    def line(self, x=None, y=None, xlabel = None, ylabel = None, title = None, **kwargs ):
        self._df.evaluate().line(x=x, y=y, xlabel=xlabel, ylabel=ylabel, title=title, df=self, **kwargs)
    
    def scatter(self, x=None, y=None, xlabel = None, ylabel = None, title = None, **kwargs ):
        self._df.evaluate().scatter(x=x, y=y, xlabel=xlabel, ylabel=ylabel, title=title, df=self, **kwargs)
    
    def scatter2d_class(self, x1=None, x2=None, y=None, xlabel=None, ylabel=None, title=None, loc='upper right', noise=0, **kwargs):
        self._df.evaluate().scatter2d_class(x1=x1, x2=x2, y=y, xlabel=xlabel, ylabel=ylabel, title=title, loc=loc, noise=noise, df=self, **kwargs)

    def scatter2d_color(self, x1=None, x2=None, c=None, xlabel=None, ylabel=None, title=None, loc='upper right', noise=0, **kwargs):
        self._df.evaluate().scatter2d_color(x1=x1, x2=x2, c=c, xlabel=xlabel, ylabel=ylabel, title=title, loc=loc, noise=noise, df=self, **kwargs)

    def scatter2d_size(self, x1=None, x2=None, s=None, xlabel=None, ylabel=None, title=None, loc='upper right', noise=0, **kwargs):
        self._df.evaluate().scatter2d_size(x1=x1, x2=x2, s=s, xlabel=xlabel, ylabel=ylabel, title=title, loc=loc, noise=noise, df=self, **kwargs)

    def plot_boundary(self, predict):
        self._df.evaluate().plot_boundary(predict)
        
    def plot_contour(self, predict):
        self._df.evaluate().plot_contour(predict)