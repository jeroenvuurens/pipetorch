import numpy as np
import pandas as pd
from pandas.core.groupby.generic import DataFrameGroupBy, SeriesGroupBy
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
import matplotlib.pyplot as plt
from sklearn.utils import resample
import copy
import os

def to_numpy(arr):
    try:
        return arr.data.cpu().numpy()
    except: pass
    try:
        return arr.to_numpy()
    except: pass
    return arr

class PTDS:
    _metadata = ['_df', '_dfindices', '_pt_categoryx', '_pt_categoryy', '_pt_dummiesx', '_pt_dummiesy', '_pt_columny', '_pt_columnx', '_pt_transposey', '_pt_bias', '_pt_polynomials', '_pt_dtype', '_pt_sequence_window', '_pt_sequence_shift_y', '_pt_is_test']
    _internal_names = pd.DataFrame._internal_names + ["_pt__indices", "_pt__x_sequence"]
    _internal_names_set = set(_internal_names)
    
    def to_ptdataframe(self):
        cls = self._df.__class__
        r = cls(self)

        r._pt_columnx = self._pt_columnx
        r._pt_columny = self._pt_columny
        r._pt_transposey = self._pt_transposey
        r._pt_bias = self._pt_bias
        r._pt_polynomials = self._pt_polynomials
        r._pt_sequence_window = self._pt_sequence_window
        r._pt_sequence_shift_y = self._pt_sequence_shift_y
        
        r._pt__train = self
        r._pt__full = self
        r._pt__valid = None
        r._pt__test = None
        r._pt_indices = list(range(len(self)))
        r._pt__train_indices = r._pt_indices
        r._pt__valid_indices = []
        r._pt__test_indices = []
        r._pt_split = None
        r._pt_random_state = None
        r._pt_balance = None
        r._pt_shuffle = False
        return r

    def _copy_meta(self, r):
        r._df = self._df
        r._dfindices = self._dfindices
        r._pt_categoryx = self._pt_categoryx
        r._pt_categoryy = self._pt_categoryy
        r._pt_dummiesx = self._pt_dummiesx
        r._pt_dummiesy = self._pt_dummiesy
        r._pt_columny = self._pt_columny
        r._pt_columnx = self._pt_columnx
        r._pt_is_test = self._pt_is_test
        r._pt_transposey = self._pt_transposey
        r._pt_polynomials = self._pt_polynomials
        r._pt_bias = self._pt_bias
        r._pt_dtype = self._pt_dtype
        r._pt_sequence_window = self._pt_sequence_window
        r._pt_sequence_shift_y = self._pt_sequence_shift_y
        return r
    
    def _ptdataset(self, data):
        return self._copy_meta( PTDataSet(data) )
    
    def _not_nan(self, a):
        a = np.isnan(a)
        while len(a.shape) > 1:
            a = np.any(a, -1)
        return np.where(~a)[0]
    
    @property
    def _dtype(self):
        return self._pt_dtype
    
    @property
    def indices(self):
        try:
            return self._pt__indices
        except:
            if self._pt_is_test:
                self._pt__indices = self._not_nan(self._x_sequence)
            else:
                s = set(self._not_nan(self._y_transposed))
                self._pt__indices = [ i for i in self._not_nan(self._x_sequence) if i in s]
            return self._pt__indices
    
    @property
    def _scalerx(self):
        return self._df._scalerx
        
    @property
    def _scalery(self):
        return self._df._scalery

    @property
    def _categoryx(self):
        return self._pt_categoryx()
        
    @property
    def _categoryy(self):
        return self._pt_categoryy()

    @property
    def _dummiesx(self):
        return self._pt_dummiesx()
        
    @property
    def _dummiesy(self):
        return self._pt_dummiesy()

    @property
    def _shift_y(self):
        if self._pt_sequence_shift_y is not None:
            return self._pt_sequence_shift_y
        else:
            return 0
    
    @property
    def _sequence_window(self):
        try:
            if self._is_sequence:
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

    def resample_rows(self, n=True):
        r = self._ptdataset(self)
        if n == True:
            n = len(r)
        if n < 1:
            n = n * len(r)
        return r.iloc[resample(list(range(len(r))), n_samples = int(n))]
    
    def interpolate_factor(self, factor=2, sortcolumn=None):
        if not sortcolumn:
            sortcolumn = self.columns[0]
        df = self.sort_values(by=sortcolumn)
        for i in range(factor):
            i = df.rolling(2).sum()[1:] / 2.0
            df = pd.concat([df, i], axis=0)
            df = df.sort_values(by=sortcolumn)
        return self._df._ptdataset(df).reset_index(drop=True)
    
    @property
    def _x_category(self):
        if self._is_sequence:
            self = self.iloc[:-self._shift_y]
        if self._categoryx is None:
            return self[self._columnx]
        r = copy.copy(self[self._columnx])
        for c, cat in zip(r._columnx, r._categoryx):
            if cat is not None:
                r[c] = cat.transform(r[c])
        return r
    
    @property
    def _x_dummies(self):
        if self._dummiesx is None:
            return self._x_category
        r = copy.copy(self._x_category)
        r1 = []
        for d, onehot in zip(r._columnx, r._dummiesx):
            if onehot is not None:
                a = onehot.transform(r[[d]])
                r1.append( pd.DataFrame(a.toarray(), columns=onehot.get_feature_names_out([d])) )
                r = r.drop(columns = d)
        r1.insert(0, r.reset_index(drop=True))
        r = pd.concat(r1, axis=1)
        return r
    
    @property
    def _x_numpy(self):
        return self._x_dummies.to_numpy()
    
    @property
    def _x_polynomials(self):
        try:
            return self._polynomials.fit_transform(self._x_numpy)
        except:
            return self._x_numpy

    @property
    def _x_scaled(self):
        if len(self) > 0:
            return self._transform(self._scalerx, self._x_polynomials)
        return self._x_polynomials
            
    @property
    def _x_biased(self):
        a = self._x_scaled
        if self._bias:
            return np.concatenate([np.ones((len(a),1)), a], axis=1)
        return a
    
    @property
    def _x_sequence(self):
        try:
            return self._pt__x_sequence
        except:
            if not self._is_sequence:
                self._pt__x_sequence = self._x_biased
            else:
                X = self._x_biased
                window = self._sequence_window
                len_seq_mode = max(0, len(X) - window + 1)
                self._pt__x_sequence =  np.concatenate([np.expand_dims(X[ii:ii+window], axis=0) for ii in range(len_seq_mode)], axis=0)        
            return self._pt__x_sequence
    
    @property
    def X(self):
        return self._x_sequence[self.indices]

    @property
    def X_tensor(self):
        import torch
        if self._dtype is None:
            return torch.tensor(self.X).type(torch.FloatTensor)
        else:
            return torch.tensor(self.X)

    @property
    def y_tensor(self):
        import torch
        if self._dtype is None:
            return torch.tensor(self.y).type(torch.FloatTensor)
        else:
            return torch.tensor(self.y)
 
    @property
    def _is_sequence(self):
        return self._pt_sequence_window is not None
        
    @property
    def tensors(self):
        return self.X_tensor, self.y_tensor
            
    @property
    def _range_y(self):
        stop = len(self) if self._shift_y >= 0 else len(self) + self._shift_y
        start = min(stop, self._sequence_window + self._shift_y - 1)
        return slice(start, stop)
        
    @property
    def _y_category(self):
        if self._is_sequence:
            self = self.iloc[self._range_y]
        if self._categoryy is None:
            return self[self._columny]
        r = copy.copy(self[self._columny])
        for d, onehot in zip(r._columny, r._dummiesy):
            if onehot is not None:
                r[c] = cat.transform(r[c])
        return r
    
    @property
    def _y_dummies(self):
        if self._dummiesy is None:
            return self._y_category
        r = copy.copy(self._y_category)
        r1 = []
        for d, onehot in zip(r._columny, r._dummiesy):
            if onehot is not None:
                a = onehot.transform(r[[d]])
                r1.append( pd.DataFrame(a.toarray(), columns=onehot.get_feature_names_out([d])) )
                r = r.drop(columns = d)
        r1.insert(0, r.reset_index(drop=True))
        r = pd.concat(r1, axis=1)
        return r
    
    @property
    def _y_numpy(self):
        return self._y_dummies.to_numpy()
    
    @property
    def _y_scaled(self):
        if len(self) > 0:
            return self._transform(self._scalery, self._y_numpy)
        return self._y_numpy
            
    @property
    def _y_transposed(self):
        return self._y_scaled.squeeze() if self._transposey else self._y_scaled
    
    @property
    def y(self):
        return self._y_transposed[self.indices]
    
    def replace_y(self, new_y):
        y_pred = self._predict(new_y)
        offset = self._range_y.start
        indices = [ i + offset for i in self.indices ]
        assert len(y_pred) == len(indices), f'The number of predictions ({len(y_pred)}) does not match the number of samples ({len(indices)})'
        r = copy.deepcopy(self)
        r[self._columny] = np.NaN
        columns = [r.columns.get_loc(c) for c in self._columny]
        r.iloc[indices, columns] = y_pred.values
        return r
    
    def to_dataset(self):
        """
        returns: a list with a train, valid and test DataSet. Every DataSet contains an X and y, where the 
        input data matrix X contains all columns but the last, and the target y contains the last column
        columns: list of columns to convert, the last column is always the target. default=None means all columns.
        """
        from torch.utils.data import TensorDataset
        return TensorDataset(*self.tensors)
    
    def _predict_y(self, predict):
        if not callable(predict):
            return predict
        try:
            from torch import nn
            import torch
            with torch.set_grad_enabled(False):
                return to_numpy(predict(self.X_tensor)).reshape(len(self))
        except:
            raise
        try:
            return predict(self.X).reshape(len(self))
        except:
            raise
            raise ValueError('predict mus be a function that works on Numpy arrays or PyTorch tensors')

    def _predict(self, predict):
        return self.inverse_transform_y(self._predict_y(predict))

    def predict(self, predict, drop=True):
        y_pred = self._predict_y(predict)
        if drop:
            return self._df.inverse_transform(self.X, y_pred)
        return self._df.inverse_transform(self.X, self.y, y_pred)
    
    def add_column(self, y_pred, *columns):
        y_pred = to_numpy(y_pred)
        offset = self._range_y.start
        indices = [ i + offset for i in self.indices ]

        assert len(y_pred) == len(indices), f'The number of predictions ({len(y_pred)}) does not match the number of samples ({len(indices)})'
        r = copy.deepcopy(self)
        y_pred = self.inverse_transform_y(y_pred)
        if len(columns) == 0:
            columns = [ c + '_pred' for c in self._columny ]
        for c in columns:
            r[c] = np.NaN
        columns = [r.columns.get_loc(c) for c in columns]
        r.iloc[indices, columns] = y_pred.values
        return r

    def inverse_transform_y(self, y_pred):
        return self._df.inverse_transform_y(y_pred)
    
    def line(self, x=None, y=None, xlabel = None, ylabel = None, title = None, **kwargs ):
        self._df.evaluate().line(x=x, y=y, xlabel=xlabel, ylabel=ylabel, title=title, df=self, **kwargs)
    
    def scatter(self, x=None, y=None, xlabel = None, ylabel = None, title = None, **kwargs ):
        self._df.evaluate().scatter(x=x, y=y, xlabel=xlabel, ylabel=ylabel, title=title, df=self, **kwargs)
    
    def scatter2d_class(self, x1=None, x2=None, y=None, xlabel=None, ylabel=None, title=None, loc='upper right', noise=0, **kwargs):
        self._df.evaluate().scatter2d_class(x1=x1, x2=x2, y=y, xlabel=xlabel, ylabel=ylabel, title=title, loc=loc, noise=noise, df=self, **kwargs)

    def scatter2d_color(self, x1=None, x2=None, c=None, xlabel=None, ylabel=None, title=None, noise=0, **kwargs):
        self._df.evaluate().scatter2d_color(x1=x1, x2=x2, c=c, xlabel=xlabel, ylabel=ylabel, title=title, noise=noise, df=self, **kwargs)

    def scatter2d_size(self, x1=None, x2=None, s=None, xlabel=None, ylabel=None, title=None, noise=0, **kwargs):
        self._df.evaluate().scatter2d_size(x1=x1, x2=x2, s=s, xlabel=xlabel, ylabel=ylabel, title=title, noise=noise, df=self, **kwargs)

    def plot_boundary(self, predict):
        self._df.evaluate().plot_boundary(predict)
        
    def plot_contour(self, predict):
        self._df.evaluate().plot_contour(predict)

class PTDataSet(pd.DataFrame, PTDS):
    _metadata = PTDS._metadata
    _internal_names = PTDS._internal_names
    _internal_names_set = PTDS._internal_names_set
    
    @property
    def _constructor(self):
        return PTDataSet
           
    @classmethod
    def from_ptdataframe(cls, data, df, dfindices):
        r = cls(data)
        r._df = df
        r._dfindices = dfindices
        r._pt_categoryx = df._categoryx
        r._pt_categoryy = df._categoryy
        r._pt_dummiesx = df._dummiesx
        r._pt_dummiesy = df._dummiesy
        r._pt_columny = df._columny
        r._pt_columnx = df._columnx
        r._pt_transposey = df._transposey
        r._pt_polynomials = df._pt_polynomials
        r._pt_bias = df._pt_bias
        r._pt_dtype = df._pt_dtype
        r._pt_is_test = False
        r._pt_sequence_window = df._pt_sequence_window
        r._pt_sequence_shift_y = df._pt_sequence_shift_y
        return r
    
    @classmethod
    def df_to_testset(cls, data, df, dfindices):
        r = cls.from_ptdataframe(data, df, dfindices)
        r._pt_is_test = True
        return r
    
    def groupby(self, by, axis=0, level=None, as_index=True, sort=True, group_keys=True, observed=False, dropna=True):
        r = super().groupby(by, axis=axis, level=level, as_index=as_index, sort=sort, group_keys=group_keys, observed=observed, dropna=dropna)
        return self._copy_meta( PTGroupedDataSet(r) )

class PTGroupedDataSetSeries(SeriesGroupBy, PTDS):
    _metadata = PTDS._metadata
    #_internal_names = PTDS._internal_names
    #_internal_names_set = PTDS._internal_names_set

    @property
    def _constructor(self):
        return PTGroupedDataSetSeries
    
    @property
    def _constructor_expanddim(self):
        return PTGroupedDataFrame
    
class PTGroupedDataSet(DataFrameGroupBy, PTDS):
    _metadata = PTDS._metadata
    #_internal_names = PTDS._internal_names
    #_internal_names_set = PTDS._internal_names_set

    def __init__(self, data=None):
        super().__init__(obj=data.obj, keys=data.keys, axis=data.axis, level=data.level, grouper=data.grouper, exclusions=data.exclusions,
                selection=data._selection, as_index=data.as_index, sort=data.sort, group_keys=data.group_keys,
                observed=data.observed, mutated=data.mutated, dropna=data.dropna)

    @property
    def _constructor(self):
        return PTGroupedDataSet
    
    @property
    def _constructor_sliced(self):
        return PTGroupedDataSetSeries
    
    def __iter__(self):
        for group, subset in super().__iter__():
            yield group, self._copy_meta(subset)
    
    def astype(self, dtype, copy=True, errors='raise'):
        PTDataSet.astype(self, dtype, copy=copy, errors=errors)

    def get_group(self, name, obj=None):
        return self._ptdataset( super().get_group(name, obj=obj) )
        
    def to_dataset(self):
        from torch.utils.data import ConcatDataset
        dss = []
        for key, group in self:
            dss.append( group.to_dataset())

        return ConcatDataset(dss)
