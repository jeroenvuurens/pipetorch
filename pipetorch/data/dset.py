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
    return arr

class _DSet:
    _metadata = ['_df', '_dfindices', '_pt_categoryx', '_pt_categoryy', '_pt_dummiesx', '_pt_dummiesy', 
                 '_pt_columny', '_pt_columnx', '_pt_vectory', '_pt_bias', '_pt_polynomials', 
                 '_pt_dtype', '_pt_sequence_window', '_pt_sequence_shift_y', '_pt_is_test', 
                 '_pt_dataset', '_pt_transforms']
    _internal_names = pd.DataFrame._internal_names + ["_pt__indices", "_pt__x_sequence"]
    _internal_names_set = set(_internal_names)
    
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
        r._pt_dataset = self._pt_dataset
        r._pt_transforms = self._pt_transforms
        
        r._pt__train = self
        r._pt__valid = None
        r._pt__test = None
        r._pt_locked_indices = list(range(len(self)))
        r._pt_locked_train_indices = r._pt_locked_indices
        r._pt_locked_valid_indices = []
        r._pt_locked_test_indices = []
        r._pt_locked_categoryx = self._pt_categoryx
        r._pt_locked_categoryy = self._pt_categoryy
        r._pt_locked_dummiesx = self._pt_dummiesx
        r._pt_locked_dummiesy = self._pt_dummiesy
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
        r._pt_vectory = self._pt_vectory
        r._pt_polynomials = self._pt_polynomials
        r._pt_transforms = self._pt_transforms
        r._pt_dataset = self._pt_dataset
        r._pt_bias = self._pt_bias
        r._pt_dtype = self._pt_dtype
        r._pt_sequence_window = self._pt_sequence_window
        r._pt_sequence_shift_y = self._pt_sequence_shift_y
        return r
    
    def _dset(self, data):
        return self._copy_meta( DSet(data) )
    
    def _not_nan(self, a):
        a = pd.isnull(a)
        while len(a.shape) > 1:
            a = np.any(a, -1)
        return np.where(~a)[0]
    
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
        return self.df_to_dset(df).to_dataset()    

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
    def _vectory(self):
        return True if self._pt_vectory is None else self._pt_vectory
            
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
        if self._pt_dtype is None:
            return torch.tensor(self.y).type(torch.FloatTensor)
        if self._pt_dtype and np.issubdtype(self._pt_dtype, np.number):
            return torch.tensor(self.y.astype(self._pt_dtype))
        else:
            return torch.tensor(self.y)
 
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
        return self._y_scaled.squeeze() if self._vectory else self._y_scaled
    
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
        offset = self._range_y.start
        indices = [ i + offset for i in self.indices ]
        assert len(y_pred) == len(indices), f'The number of predictions ({len(y_pred)}) does not match the number of samples ({len(indices)})'
        r = copy.deepcopy(self)
        r[self._columny] = np.NaN
        columns = [r.columns.get_loc(c) for c in self._columny]
        r.iloc[indices, columns] = y_pred.values
        return r
    
    def to_dataset(self, dataset=None):
        """
        Converts this DSet into a PyTorch DataSet.
        
        Arguments: 
            dataset: class (TensorDataset)
                the class to use to instantiate the dataset
        
        returns: DataSet
            A PyTorch DataSet over X_tensor and y_tensor
        """
        self._pt_dataset = dataset or self._pt_dataset
        if self._pt_dataset is not None:
            r = self._pt_dataset(*self.tensors)
        elif self._pt_dtype is str or self._pt_dtype == False:
            r = NumpyDataset(self.X, self.y)
        else:
            from torch.utils.data import TensorDataset
            r = TensorDataset(*self.tensors)
        if self._pt_transforms is not None:
            r = TransformableDataset(r, self._pt_dtype, *self._pt_transforms)
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
    
    def add_column(self, y_pred, *columns):
        y_pred = to_numpy(y_pred)
        offset = self._range_y.start
        indices = [ i + offset for i in self.indices ]

        assert len(y_pred) == len(indices), f'The number of predictions ({len(y_pred)}) does not match the number of samples ({len(indices)})'
        r = copy.deepcopy(self)
        y_pred = self.inverse_scale_y(y_pred)
        if len(columns) == 0:
            columns = [ c + '_pred' for c in self._columny ]
        for c in columns:
            r[c] = np.NaN
        columns = [r.columns.get_loc(c) for c in columns]
        r.iloc[indices, columns] = y_pred.values
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
        
        self._df.evaluate().line(x=x, y=y, xlabel=xlabel, ylabel=ylabel, title=title, df=self, **kwargs)
    
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
        self._df.evaluate().scatter(x=x, y=y, xlabel=xlabel, ylabel=ylabel, title=title, df=self, **kwargs)
    
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
        self._df.evaluate().scatter2d_class(x1=x1, x2=x2, y=y, xlabel=xlabel, ylabel=ylabel, title=title, loc=loc, noise=noise, df=self, **kwargs)

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
        self._df.evaluate().scatter2d_color(x1=x1, x2=x2, c=c, xlabel=xlabel, ylabel=ylabel, title=title, noise=noise, df=self, **kwargs)

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
        self._df.evaluate().scatter2d_size(x1=x1, x2=x2, s=s, xlabel=xlabel, ylabel=ylabel, title=title, noise=noise, df=self, **kwargs)

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
        self._df.evaluate().plot_boundary(predict, levels=levels)
        
class DSet(pd.DataFrame, _DSet):
    _metadata = _DSet._metadata
    _internal_names = _DSet._internal_names
    _internal_names_set = _DSet._internal_names_set
    
    @property
    def _constructor(self):
        return DSet
           
    @classmethod
    def from_dframe(cls, data, df, dfindices, transforms):
        r = cls(data)
        r._df = df
        r._dfindices = dfindices
        r._pt_categoryx = df._categoryx
        r._pt_categoryy = df._categoryy
        r._pt_dummiesx = df._dummiesx
        r._pt_dummiesy = df._dummiesy
        r._pt_columny = df._columny
        r._pt_columnx = df._columnx
        r._pt_dataset = df._pt_dataset
        r._pt_transforms = transforms
        r._pt_vectory = df._pt_vectory
        r._pt_polynomials = df._pt_polynomials
        r._pt_bias = df._pt_bias
        r._pt_dtype = df._pt_dtype
        r._pt_is_test = False
        r._pt_sequence_window = df._pt_sequence_window
        r._pt_sequence_shift_y = df._pt_sequence_shift_y
        return r
    
    @classmethod
    def df_to_testset(cls, data, df, dfindices, transforms):
        r = cls.from_dframe(data, df, dfindices, transforms)
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
