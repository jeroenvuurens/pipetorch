import numpy as np
import pandas as pd
from pandas.core.groupby.generic import DataFrameGroupBy, SeriesGroupBy
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from ..data.transformabledataset import TransformableDataset
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
    try:
        return arr.values
    except: pass
    return arr

class _DSet:
    _metadata = ['_df', '_pt_categoryx', '_pt_categoryy', 
                 '_pt_dummiesx', '_pt_dummiesy',
                 '_pt_scalerx', '_pt_scalery', 
                 '_pt_columntransformerx', '_pt_columntransformery',
                 '_pt_columny', '_pt_columnx', 
                 '_pt_vectory', '_pt_bias', '_pt_polynomials', 
                 '_pt_dtype', 
                 '_pt_sequence_window', '_pt_sequence_shift_y', '_pt_is_test', 
                 '_pt_datasetclass', '_pt_dataset_transforms', '_pt_filterna']
    _internal_names = pd.DataFrame._internal_names + ["_pt__indices", "_pt__x_sequence"]
    _internal_names_set = set(_internal_names)
    
    def __init__(self, data, **kwargs):
        for m in self._metadata:
            self.__setattr__(m, None)
            
    def to_dframe(self):
        cls = self._df.__class__
        r = cls(self)

        r._pt_columnx = self._pt_columnx
        r._pt_columny = self._pt_columny
        r._pt_vectory = self._pt_vectory
        r._pt_bias = self._pt_bias
        r._pt_polynomials = self._pt_polynomials
        r._pt_sequence_window = self._pt_sequence_window
        r._pt_sequence_shift_y = self._pt_sequence_shift_y
        r._pt_filterna = self._pt_filterna
        r._pt_dataset_transforms = self._pt_dataset_transforms
        
        r._pt__train = self
        r._pt__valid = None
        r._pt__test = None
        r._categoryx = self._categoryx
        r._categoryy = self._categoryy
        r._dummiesx = self._dummiesx
        r._dummiesy = self._dummiesy
        r._scalerx = self._scalerx
        r._scalery = self._scalery
        r._pt_split = None
        r._pt_random_state = None
        r._pt_balance = None
        r._pt_shuffle = False
        return r

    def _copy_meta(self, r):
        r._df = self._df
        r._pt_categoryx = self._categoryx
        r._pt_categoryy = self._categoryy
        r._pt_dummiesx = self._dummiesx
        r._pt_dummiesy = self._dummiesy
        r._pt_scalerx = self._scalerx
        r._pt_scalery = self._scalery
        r._pt_columny = self._pt_columny
        r._pt_columnx = self._pt_columnx
        r._pt_columntransformery = self._columntransformery
        r._pt_columntransformerx = self._columntransformerx
        r._pt_is_test = self._pt_is_test
        r._pt_vectory = self._pt_vectory
        r._pt_polynomials = self._pt_polynomials
        r._pt_dataset_transforms = self._pt_dataset_transforms
        r._pt_datasetclass = self._pt_datasetclass
        r._pt_bias = self._pt_bias
        r._pt_dtype = self._pt_dtype
        r._pt_sequence_window = self._pt_sequence_window
        r._pt_sequence_shift_y = self._pt_sequence_shift_y
        r._pt_filterna = self._pt_filterna
        return r
    
    def _dset(self, data):
        return self._copy_meta( DSet(data) )
    
    def df_to_dset(self, df):
        """
        Converts a DataFrame to a DSet that has the pipeline as this DSet.
        
        Arguments:
            df: DataFrame
        
        Returns: DSet
        """
        return self._df.df_to_dset(df)

    def df_to_dataset(self, df):
        """
        Converts the given df to a DataSet using the pipeline of this DSet.
        
        Arguments:
            df: DataFrame or DFrame
                to convert into a DataSet
        
        returns: DataSet.
        """
        return self.df_to_dset(df).to_dataset(self._pt_datasetclass)    

    @property
    def _dtype(self):
        return self._pt_dtype
    
    def _not_nan_y_transposed(self):
        y = self._y_transposed
        if len(y.shape) == 1:
            mask = np.isnan(self._y_transposed)
        else:
            mask = np.any(np.isnan(self._y_transposed))
        return ~mask

    def _not_nan_x_sequence(self):
        mask = np.any(np.isnan(self._x_sequence), axis=1)
        return ~mask

    def _not_nan_x_numpy(self):
        mask = np.any(np.isnan(self._x_numpy), axis=1)
        return ~mask
    
    @property
    def indices(self):
        try:
            return self._pt__indices
        except:
            if self._pt_filterna:
                if self._pt_is_test:
                    if self._is_sequence:
                        self._pt__indices = self._not_nan_x_sequence()
                    else:
                        self._pt__indices = self._not_nan_x_numpy()
                else:
                    if self._is_sequence:
                        self._pt__indices = self._not_nan_y_transposed() & self._not_nan_x_sequence()
                    else:
                        self._pt__indices = self._not_nan_y_transposed() & self._not_nan_x_numpy()
            else:
                self._pt__indices = [ True ] * len(self._y_transposed)
            return self._pt__indices
        
    @property
    def length(self):
        return sum(self.indices)
    
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
    def _start_index_y(self):
        if self._is_sequence:
            return self._pt_sequence_window+self._shift_y-1
        else:
            return 0

    @property
    def _columny(self):
        return [ self.columns[-1] ] if self._pt_columny is None else self._pt_columny
        
    @property
    def _vectory(self):
        return True if self._pt_vectory is None else self._pt_vectory
            
    @property
    def _columnx(self):
        if self._pt_columnx is None:
            return [ c for c in self.columns if c not in self._columny ]
        return self._pt_columnx
   
    @property
    def _scalerx(self):
        if self._pt_scalerx is not None:
            return self._pt_scalerx
        self._pt_scalerx = self._df._scalerx
        return self._pt_scalerx

    @_scalerx.setter
    def _scalerx(self, value):
        self._pt_scalerx = value

    @property
    def _scalery(self):
        if self._pt_scalery is not None:
            return self._pt_scalery
        self._pt_scalery = self._df._scalery
        return self._pt_scalery

    @_scalery.setter
    def _scalery(self, value):
        self._pt_scalery = value

    @property
    def _categoryx(self):
        if self._pt_categoryx is not None:
            return self._pt_categoryx
        self._pt_categoryx = self._df._categoryx()
        return self._pt_categoryx

    @_categoryx.setter
    def _categoryx(self, value):
        self._pt_categoryx = value

    @property
    def _categoryy(self):
        if self._pt_categoryy is not None:
            return self._pt_categoryy
        self._pt_categoryy = self._df._categoryy()
        return self._pt_categoryy

    @_categoryy.setter
    def _categoryy(self, value):
        self._pt_categoryy = value

    @property
    def _columntransformerx(self):
        if self._pt_columntransformerx is not None:
            return self._pt_columntransformerx
        self._pt_columntransformerx = self._df._columntransformerx()
        return self._pt_columntransformerx

    @_columntransformerx.setter
    def _columntransformerx(self, value):
        self._pt_columntransformerx = value

    @property
    def _columntransformery(self):
        if self._pt_columntransformery is not None:
            return self._pt_columntransformery
        self._pt_columntransformery = self._df._columntransformery()
        return self._pt_columntransformery

    @_columntransformery.setter
    def _columntransformery(self, value):
        self._pt_columntransformery = value

    @property
    def _dummiesx(self):
        if self._pt_dummiesx is not None:
            return self._pt_dummiesx
        self._pt_dummiesx = self._df._dummiesx()
        return self._pt_dummiesx

    @_dummiesx.setter
    def _dummiesx(self, value):
        self._pt_dummiesx = value

    @property
    def _dummiesy(self):
        if self._pt_dummiesy is not None:
            return self._pt_dummiesy
        self._pt_dummiesy = self._df._dummiesy()
        return self._pt_dummiesy

    @_dummiesy.setter
    def _dummiesy(self, value):
        self._pt_dummiesy = value

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
        """
        Resamples this DSet with replacement to support bootstrapping.
        
        Arguments:
            n: int (True)
                The number of samples to take, or the size of the dataset if n equals True.
        
        Returns: DSet
            A resampled DSet
        """
        r = self._dset(self)
        if n == True:
            n = len(r)
        if n < 1:
            n = n * len(r)
        return r.iloc[resample(list(range(len(r))), n_samples = int(n))]
    
    def interpolate_factor(self, factor=2, sortcolumn=None):
        """
        Interpolates the data between the rows. This function was created to simplify drawing
        of higher order polynomial functions, and may be useful for other situations, but it is
        limited to simple interplation between two consecutive points.
        
        Arguments:
            factor: int (2)
                interpolates by ading 2 ** factor - 1 values. In other words, a factor of 1 interpolates
                1 value betweem every 2 existing values and a factor of 2 will interpolate 3 values between 
                every two values.
            sortcolumn: str (None)
                Before interpolating, the DSet is first sorted by this column, or the first column if None is
                provided.
                
        Returns: DSet
            In which the values are interpolated.
        """
        if not sortcolumn:
            sortcolumn = self.columns[0]
        df = self.sort_values(by=sortcolumn)
        for i in range(factor):
            i = df.rolling(2).sum()[1:] / 2.0
            df = pd.concat([df, i], axis=0)
            df = df.sort_values(by=sortcolumn)
        return self._df._dset(df).reset_index(drop=True)
    
    @property
    def _x_transformed(self):
        if self._is_sequence:
            self = self.iloc[:-self._shift_y]
        if self._columntransformerx is None:
            return self[self._columnx]
        r = copy.copy(self[self._columnx])
        for c, t in zip(r._columnx, r._columntransformerx):
            if t is not None:
                r[c] = t.transform(r[c])
        return r
    
    @property
    def _x_category(self):
        if self._categoryx is None:
            return self._x_transformed
        r = copy.copy(self._x_transformed)
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
        """
        Constructs a Numpy array in which the values for X are transformed according to the configured pipeline.
        
        Returns: Numpy array
        """
        return self._x_sequence[self.indices]

    @property
    def X_tensor(self):
        """
        Constructs a PyTorch tensor in which the values for X are transformed according to the configured pipeline.
        A difference with X() is that the data is transformed to a single datatype (by default torch.FloatTensor).
        
        Returns: PyTorch tensor
        """
        
        import torch
        if self._pt_dtype is None:
            return torch.tensor(self.X).type(torch.FloatTensor)
        if self._pt_dtype and np.issubdtype(self._pt_dtype, np.number):
            return torch.tensor(self.X.astype(self._pt_dtype))
        return torch.tensor(self.X)

    @property
    def y_tensor(self):
        """
        Constructs a PyTorch tensor in which the values for y are transformed according to the configured pipeline.
        A difference with y() is that the data is transformed to a single datatype (by default torch.FloatTensor).
        
        Returns: PyTorch tensor
        """

        import torch
        y = self._y_scaled[self.indices] if self._pt_vectory is None else self._y
        if self._pt_dtype is None:
            return torch.tensor(y).type(torch.FloatTensor)
        if self._pt_dtype and np.issubdtype(self._pt_dtype, np.number):
            return torch.tensor(y.astype(self._pt_dtype))
        else:
            return torch.tensor(y)
 
    @property
    def _is_sequence(self):
        return self._pt_sequence_window is not None
        
    @property
    def tensors(self):
        """
        Combines X_tensor() and y_tensor()
        
        Returns: tensor, tensor
        """

        return self.X_tensor, self.y_tensor
            
    @property
    def _range_y(self):
        stop = len(self) if self._is_sequence and self._shift_y >= 0 else len(self) + self._shift_y
        start = min(stop, self._start_index_y)
        return slice(start, stop)

    @property
    def _y_transformed(self):
        if self._is_sequence:
            self = self.iloc[self._range_y]
        if self._categoryy is None:
            return self[self._columny]
        if self._columntransformery is None:
            return self[self._columny]
        r = copy.copy(self[self._columny])
        for c, t in zip(r._columny, r._columntransformery):
            if t is not None:
                r[c] = t.transform(r[c])
        return r
    
    @property
    def _y_category(self):
        if self._categoryy is None:
            return self._y_transformed
        r = copy.copy(self._y_transformed)
        for c, cat in zip(r._columny, r._categoryy):
            if cat is not None:
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
        return self._y_scaled.squeeze(axis=1) if self._vectory else self._y_scaled
    
    @property
    def y(self):
        """
        Constructs a Numpy array in which the values for y are transformed according to the configured pipeline.
        
        Returns: Numpy array
        """
        
        return self._y_transposed[self.indices]
        
    def replace_y(self, new_y):
        """
        Returns a copy of this DSet, in which y is replaced with alternative values.
        
        Arguments:
            new_y: Numpy array, PyTorch tensor or callable
                If new_y is a callable, it will be called with this DSet's X to generate predicted values of y
        
        Returns: DSet
            A copy of this DSet, in which the values for y are replaced by the given values new_y
        """
        
        y_pred = self._predict(new_y)
        indices = self.indices
        offset = self._range_y.start
        if offset > 0:
            indices = [False] * offset + indices[:len(self) - offset]
        assert len(y_pred) == sum(indices), f'The number of predictions ({len(y_pred)}) does not match the number of samples ({len(indices)})'
        r = copy.deepcopy(self)
        columns = [r.columns.get_loc(c) for c in self._columny]
        #r.iloc[indices, columns] = np.NaN
        r.iloc[indices, columns] = to_numpy(y_pred)
        return r
    
    def to_dataset(self, datasetclass=None):
        """
        Converts this DSet into a PyTorch DataSet.
        
        Arguments: 
            datasetclass: class (TensorDataset)
                the class to use to instantiate the dataset
        
        returns: DataSet
            A PyTorch DataSet over X_tensor and y_tensor
        """
        self._pt_datasetclass = datasetclass or self._pt_datasetclass
        if self._pt_datasetclass is not None:
            r = self._pt_datasetclass(*self.tensors)
        elif self._pt_dtype is str or self._pt_dtype == False:
            r = NumpyDataset(self.X, self.y)
        else:
            from torch.utils.data import TensorDataset
            r = TensorDataset(*self.tensors)
        if self._pt_dataset_transforms is not None:
            r = TransformableDataset(r, self._pt_dtype, *self._pt_dataset_transforms)
        return r
    
    def to_dataloader(self, batch_size=32, shuffle=True, collate_fn=None, **kwargs):
        """
        Converts this DSet into a PyTorch DataLoader.
        
        Arguments: 
            batch_size: int (32)
            
            shuffle: bool (True)
            
            collate_fn: callable (None)
            
            **kwargs: dict
                passed on to DataLoader
        
        returns: DataLoader
            A PyTorch DataLoader over X_tensor and y_tensor
        """
        if collate_fn is not None:
            kwargs['collate_fn'] = collate_fn
        from torch.utils.data import DataLoader
        return DataLoader(self.to_dataset(), batch_size=batch_size, shuffle=shuffle, **kwargs)
    
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
        return self.inverse_scale_y(self._predict_y(predict))

    def predict(self, predict, drop=True):
        y_pred = self._predict_y(predict)
        if drop:
            return self._df.inverse_scale(self.X, y_pred)
        return self._df.inverse_scale(self.X, self.y, y_pred)
    
    def add_column(self, y_pred, *columns, inplace=False):
        """
        Fills the given predictions into the columns.
        
        Depending on inplace, True means the values are placed in the original 
        DFrame from which this subset was generated, and False means that a new
        copy of the subset (DSet) is returned. See fill_column().
        
        The predictions are automatically converted (scaled) back using 
        inverse_transform to the original scale.
        
        Args:
            y_pred: 2D Array like data
                the values to be stored in the Dataframe. The number of rows must
                match the number of values for the target variable in the subset.
            *columns: str (None)
                the names of columns in which to store the values. The number of
                columns must match the number of columns in values. If None,
                new columns are added with the name of the target variable(s) and 
                the suffix '_pred'
            inplace: bool (False)
                whether to write the values in a local copy of the subset or to
                write them in the original DFrame that generated this subset. Read
                the warnings above.
            offeset: int (0)
                offset can be set when sequences are used to set values on the same
                row as the target variable.
                
        Returns: DSet or None
            Either a DSet with modified values (inplace=False) or None (inplace=True)
        """
        
        y_pred = to_numpy(y_pred)
        offset = self._range_y.start
        y_pred = self.inverse_scale_y(y_pred)
        if len(columns) == 0:
            columns = [ c + '_pred' for c in self._columny ]
        
        return self.fill_column(y_pred, *columns, offset=offset, inplace=inplace)

    def fill_column(self, values, *columns, inplace=False, offset=0):
        """
        Fills the given columns with the given values.
        
        Depending on inplace, True means the values are placed in the original 
        DFrame from which this subset was generated, and False means that a new
        copy of the subset (DSet) is returned. 
        
        Both variants come with a caution:
        INPLACE=True means that the original DFrame will have to regenerate all subsets
        (which it will do automatically) and any copy of a subset will become invalid.
        Additionally, columns that are added will also be added to the entire DFrame
        and therefore also any subset, while only this subset will be seeded with
        values. Use this when the data is to be used for further processing with
        PipeTorch.
        
        INPLACE=False means that the changes are lost when the subsets are regenerated 
        (e.g. when you reconfigure the pipeline). Also, the connection between the
        DFrame and the subset are broken, therefore preparing data may not be possible
        anymore. Use this to add data to a subset to process with Pandas for reporting.

        Args:
            values: 2D Array like data
                the values to be stored in the Dataframe. The number of rows must
                match the number of rows in the subset.
            *columns: str
                the names of columns in which to store the values. The number of
                columns must match the number of columns in values.
            inplace: bool (False)
                whether to write the values in a local copy of the subset or to
                write them in the original DFrame that generated this subset. Read
                the warnings above.
            offeset: int (0)
                offset can be set when sequences are used to set values on the same
                row as the target variable.
                
        Returns: DSet or None
            Either a DSet with modified values (inplace=False) or None (inplace=True)
        """
        indices = self.indices
        if offset > 0:
            indices = [False] * offset + indices[:len(self)-offset]
        if np.isscalar(values):
            v = np.empty((sum(indices), len(columns)))
            v[:,:] = values
            values = v
        else:
            values = to_numpy(values)

        assert len(values) == sum(indices), f'The number of values ({len(values)}) does not match the number of samples ({sum(indices)})'
        
        if inplace:
            try:
                dfindices = self.index[indices]
            except:
                raise TypeError('You can only use fill_column inplace if an index is present.')

            for c in columns:
                if c not in self._df.columns:
                    self._df[c] = np.NaN
        #columns = [self._df.columns.get_loc(c) for c in columns]
            self._df.loc[ dfindices, columns ] = values
            self._df._columns_changed()
        else:
            r = copy.deepcopy(self)
            for c in columns:
                r[c] = np.NaN
            #columns = [r.columns.get_loc(c) for c in columns]
            r.loc[ indices, columns] = values
            return r

    def inverse_scale_y(self, y_pred):
        return self._df.inverse_scale_y(y_pred)
    
    def line(self, x=None, y=None, xlabel = None, ylabel = None, title = None, **kwargs ):
        """
        Plots a line graph of this dataset using matplotlib.
        
        Args:
            x: str (None)
                the column to use on the x-axis. If None, the first input feature is used.
                
            y: str, Array or function (None)
                the column to use in the y-axis. If None, the first target feature is used.
                You may also pass an array with values (for example model predictions), but these
                must be paired with the dataset rows. Alernatively, pass a function(X) that is called on X 
                to generate y. PipeTorch will attempt to first use a tensor (in case of a PyTorch model) 
                and when that fails with a Numpy Array.
                
            xlabel: str (None)
                the label to use on the x-axis. If None, the name of x is used.
                
            ylabel: str (none)
                the label to use on the y-axis. If None, the name of y is used.
                
            title: str (None)
                the title used for the figure
                
            kwargs: dict
                arguments that are passed to plt.plot
        """
        
        self._df._evaluator().line(x=x, y=y, xlabel=xlabel, ylabel=ylabel, title=title, df=self, **kwargs)
    
    def scatter(self, x=None, y=None, xlabel = None, ylabel = None, title = None, **kwargs ):
        """
        Plots a scatter graph of this dataset using matplotlib.
        
        Args:
            x: str (None)
                the column to use on the x-axis. If None, the first input feature is used.
                
            y: str, Array or function (None)
                the column to use in the y-axis. If None, the first target feature is used.
                You may also pass an array with values (for example model predictions), but these
                must be paired with the dataset rows. Alernatively, pass a function(X) that is called on X 
                to generate y. PipeTorch will attempt to first use a tensor (in case of a PyTorch model) 
                and when that fails with a Numpy Array.
                
            xlabel: str (None)
                the label to use on the x-axis. If None, the name of x is used.
                
            ylabel: str (none)
                the label to use on the y-axis. If None, the name of y is used.
                
            title: str (None)
                the title used for the figure
                
            kwargs: dict
                arguments that are passed to plt.plot
        """
        self._df._evaluator().scatter(x=x, y=y, xlabel=xlabel, ylabel=ylabel, title=title, df=self, **kwargs)
    
    def scatter2d_class(self, x1=None, x2=None, y=None, xlabel=None, ylabel=None, title=None, loc='upper right', noise=0, **kwargs):
        """
        Plots a 2d scatter graph of this dataset using matplotlib. The y-label is used as a class label.
        
        Args:
            x1: str (None)
                the column to use on the x-axis. If None, the first input feature is used.
                
            x2: str (None)
                the column to use on the x-axis. If None, the second input feature is used.
                
            y: str, Array or function (None)
                the column to use as the series for the plot. If None, the first target feature is used.
                You may also pass an array with values (for example model predictions), but these
                must be paired to the dataset rows. Alernatively, pass a function(X) that is called on X 
                to generate y. PipeTorch will attempt to first use a tensor (in case of a PyTorch model) 
                and when that fails with a Numpy Array.
                
            xlabel: str (None)
                the label to use on the x-axis. If None, the name of x is used.
                
            ylabel: str (none)
                the label to use on the y-axis. If None, the name of y is used.
                
            title: str (None)
                the title used for the figure
                
            loc: str ('upper right')
                passed to plt.legend to place the legend in a certain position
                
            noise: 0 (float)
                transforms s0 that x1 and x2 are incremented with noise multiplied by a random number
                from their respecrive standard deviation. This allows better visualization of discrete data.
                
            kwargs: dict
                arguments that are passed to plt.plot
        """
        self._df._evaluator().scatter2d_class(x1=x1, x2=x2, y=y, xlabel=xlabel, ylabel=ylabel, title=title, loc=loc, noise=noise, df=self, **kwargs)

    def scatter2d_color(self, x1=None, x2=None, c=None, xlabel=None, ylabel=None, title=None, noise=0, **kwargs):
        """
        Plots a 2d scatter graph of this dataset using matplotlib. The y-label is used to color the points.
        
        Args:
            x1: str (None)
                the column to use on the x-axis. If None, the first input feature is used.
                
            x2: str (None)
                the column to use on the x-axis. If None, the second input feature is used.
                
            y: str, Array or function (None)
                the column to use as the series for the plot. If None, the first target feature is used.
                You may also pass an array with values (for example model predictions), but these
                must be paired to the dataset rows. Alernatively, pass a function(X) that is called on X 
                to generate y. PipeTorch will attempt to first use a tensor (in case of a PyTorch model) 
                and when that fails with a Numpy Array.
                
            xlabel: str (None)
                the label to use on the x-axis. If None, the name of x is used.
                
            ylabel: str (none)
                the label to use on the y-axis. If None, the name of y is used.
                
            title: str (None)
                the title used for the figure
                
            loc: str ('upper right')
                passed to plt.legend to place the legend in a certain position
                
            noise: 0 (float)
                transforms s0 that x1 and x2 are incremented with noise multiplied by a random number
                from their respecrive standard deviation. This allows better visualization of discrete data.
                
            kwargs: dict
                arguments that are passed to plt.plot
        """
        self._df._evaluator().scatter2d_color(x1=x1, x2=x2, c=c, xlabel=xlabel, ylabel=ylabel, title=title, noise=noise, df=self, **kwargs)

    def scatter2d_size(self, x1=None, x2=None, s=None, xlabel=None, ylabel=None, title=None, noise=0, **kwargs):
        """
        Plots a 2d scatter graph of this dataset using matplotlib. The y-label is used as the point size.
        
        Args:
            x1: str (None)
                the column to use on the x-axis. If None, the first input feature is used.
                
            x2: str (None)
                the column to use on the x-axis. If None, the second input feature is used.
                
            y: str, Array or function (None)
                the column to use as the series for the plot. If None, the first target feature is used.
                You may also pass an array with values (for example model predictions), but these
                must be paired to the dataset rows. Alernatively, pass a function(X) that is called on X 
                to generate y. PipeTorch will attempt to first use a tensor (in case of a PyTorch model) 
                and when that fails with a Numpy Array.
                
            xlabel: str (None)
                the label to use on the x-axis. If None, the name of x is used.
                
            ylabel: str (none)
                the label to use on the y-axis. If None, the name of y is used.
                
            title: str (None)
                the title used for the figure
                
            loc: str ('upper right')
                passed to plt.legend to place the legend in a certain position
                
            noise: 0 (float)
                transforms s0 that x1 and x2 are incremented with noise multiplied by a random number
                from their respecrive standard deviation. This allows better visualization of discrete data.
                
            kwargs: dict
                arguments that are passed to plt.plot
        """
        self._df._evaluator().scatter2d_size(x1=x1, x2=x2, s=s, xlabel=xlabel, ylabel=ylabel, title=title, noise=noise, df=self, **kwargs)

    def plot_boundary(self, predict, levels=[0.5]):
        """
        Plots a decision boundary for classification models that use exactly two input features. 
        Prior to calling this function, you should already scatter_plot the dataset, beacuse this
        function uses the minimum and maximum values on the axis to do a grid search. It will then
        overlay the decision boundary on the existing plot.
        
        Args:
            predict: function (None)
                a function(X) that is called to classify an X with two features 
                PipeTorch will attempt to first use a tensor (in case of a PyTorch model) 
                and when that fails with a Numpy Array.
                
            levels: [ float ] ([0.5])
                the levels of the decision boundaries to plot. Pass multiple values or None
                to generate a contour plot.
                
            kwargs: dict
                arguments that are passed to plt.plot
        """
        self._df._evaluator().plot_boundary(predict, levels=levels)

    def inspect(self):
        """
        Describe the data in the DFrame, specifically by reporting per column 
        - Datatype 
        - Missing: number and percetage of 'Missing' values
        - Range: numeric types, are described as a range [min, max]
                  and for non-numeric types the #number of unique values is given
        - Values: the two most frequently occuring values (most frequent first).
        """

        missing_count = self.isnull().sum() # the count of missing values
        value_count = self.isnull().count() # the count of all values
        missing_percentage = round(missing_count / value_count * 100,2) #the percentage of missing values

        datatypes = self.dtypes

        df = pd.DataFrame({
            'Missing (#)': missing_count, 
            'Missing (%)': missing_percentage,
            'Datatype': datatypes
        }) #create a dataframe
        df = df.sort_values(by=['Missing (#)'], ascending=False)

        value_col = []
        range_col = []
        for index, row in df.iterrows():
            u = self[index].value_counts().index.tolist()
            if pd.api.types.is_numeric_dtype(row['Datatype']):
                _range = f"[{self[index].min()}, {self[index].max()}]"
            else:
                _range = f"#{len(u)}"
            if len(u) == 1:
                _values = f'({u})'
            elif len(u) == 2:
                _values = f'({u[0]}, {u[1]})'
            elif len(u) > 2:
                _values = f'({u[0]}, {u[1]}, ...)'
            else:
                _values = ''
            range_col.append(_range)
            value_col.append(_values)
        df["Range"] = range_col
        df["Values"] = value_col
        return df
     
class DSet(pd.DataFrame, _DSet):
    _metadata = _DSet._metadata
    _internal_names = _DSet._internal_names
    _internal_names_set = _DSet._internal_names_set
    
    def __init__(self, data, *args, **kwargs):
        super().__init__(data, *args, **kwargs)
        _DSet.__init__(self, data)
        
    @property
    def _constructor(self):
        return DSet
           
    @classmethod
    def from_dframe(cls, data, df, transforms):
        r = cls(data)
        r._df = df
        #r._dfindices = np.array(dfindices)
        r._pt_columny = df._columny
        r._pt_columnx = df._columnx
        #r._pt_dataset = df._pt_dataset
        r._pt_dataset_transforms = transforms
        r._pt_vectory = df._pt_vectory
        r._pt_polynomials = df._pt_polynomials
        r._pt_bias = df._pt_bias
        r._pt_dtype = df._pt_dtype
        r._pt_is_test = False
        r._pt_sequence_window = df._pt_sequence_window
        r._pt_sequence_shift_y = df._pt_sequence_shift_y
        r._pt_filterna = df._pt_filterna
        return r
    
    @classmethod
    def df_to_testset(cls, data, df, transforms):
        r = cls.from_dframe(data, df, transforms)
        r._pt_is_test = True
        return r
    
    def groupby(self, by, axis=0, level=None, as_index=True, sort=True, group_keys=True, observed=False, dropna=True):
        r = super().groupby(by, axis=axis, level=level, as_index=as_index, sort=sort, group_keys=group_keys, observed=observed, dropna=dropna)
        return self._copy_meta( GroupedDSet(r) )

class NumpyDataset(object):
    def __init__(self, X, y):
        self.X = X
        self.y = y

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

    def __len__(self):
        return len(self.X)
    
class GroupedDSetSeries(SeriesGroupBy, _DSet):
    _metadata = _DSet._metadata
    #_internal_names = PTDS._internal_names
    #_internal_names_set = PTDS._internal_names_set

    @property
    def _constructor(self):
        return GroupedDSetSeries
    
    @property
    def _constructor_expanddim(self):
        return GroupedDFrame
    
class GroupedDSet(DataFrameGroupBy, _DSet):
    _metadata = _DSet._metadata
    #_internal_names = PTDS._internal_names
    #_internal_names_set = PTDS._internal_names_set

    def __init__(self, data=None):
        super().__init__(obj=data.obj, keys=data.keys, axis=data.axis, level=data.level, grouper=data.grouper, exclusions=data.exclusions,
                selection=data._selection, as_index=data.as_index, sort=data.sort, group_keys=data.group_keys,
                observed=data.observed, mutated=data.mutated, dropna=data.dropna)

    @property
    def _constructor(self):
        return GroupedDSet
    
    @property
    def _constructor_sliced(self):
        return GroupedDSetSeries
    
    def __iter__(self):
        for group, subset in super().__iter__():
            yield group, selfl._copy_meta(subset)
            
    def get_group(self, name, obj=None):
        return self._dset( super().get_group(name, obj=obj) )
        
    def to_dataset(self):
        from torch.utils.data import ConcatDataset
        dss = []
        for key, group in self:
            dss.append( group.to_dataset())

        return ConcatDataset(dss)
