import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.utils import resample
import matplotlib.pyplot as plt
import copy
import path
from pathlib import Path
import os
import time
import warnings
import linecache
from .evaluate import Evaluator
from .databunch import Databunch
from .ptdataset import PTDataSet
from pandas.core.groupby.generic import DataFrameGroupBy, SeriesGroupBy

def to_numpy(arr):
    try:
        return arr.data.cpu().numpy()
    except: pass
    try:
        return arr.to_numpy()
    except: pass
    return arr

def get_filename(url):
    fragment_removed = url.split("#")[0]  # keep to left of first #
    query_string_removed = fragment_removed.split("?")[0]
    scheme_removed = query_string_removed.split("://")[-1].split(":")[-1]
    if scheme_removed.find("/") == -1:
        filename = scheme_removed
    else:
        filename = os.path.basename(scheme_removed)
    if '.' in filename:
        filename = filename.rsplit( ".", 1 )[ 0 ] + '.csv'
    return filename

def read_excel(path, filename=None, alternativesource=None, sep=None, delimiter=None, **kwargs):
    if filename is None:
        filename = get_filename(path)
    if (Path.home() / '.pipetorchuser' / filename).is_file():
        return PTDataFrame.read_csv(Path.home() / '.pipetorchuser' / filename, **kwargs)
    if (Path.home() / '.pipetorch' / filename).is_file():
        return PTDataFrame.read_csv(Path.home() / '.pipetorch' / filename, **kwargs)
    if alternativesource:
        df = pd.read_excel(alternativesource())
    else:
        print('Downloading new file ' + path)
        df = pd.read_excel(path, **kwargs)
        df.columns = df.columns.str.replace(' ', '') 
    (Path.home() / '.pipetorchuser').mkdir(exist_ok=True)
    df.to_csv(Path.home() / '.pipetorchuser' / filename, index=False)
    return PTDataFrame(df)

def read_pd_csv(path, filename=None, alternativesource=None, sep=None, delimiter=None, **kwargs):
    if sep:
        kwargs['sep'] = sep
    elif delimiter:
        kwargs['delimiter'] = delimiter
    if filename is None:
        filename = get_filename(path)
    if (Path.home() / '.pipetorchuser' / filename).is_file():
        #print(str(Path.home() / '.pipetorchuser' / filename))
        return pd.read_csv(Path.home() / '.pipetorchuser' / filename, **kwargs)
    if (Path.home() / '.pipetorch' / filename).is_file():
        #print(str(Path.home() / '.pipetorch' / filename))
        return pd.read_csv(Path.home() / '.pipetorch' / filename, **kwargs)
    if alternativesource:
        df = alternativesource()
    else:
        print('Downloading new file ' + path)
        df = pd.read_csv(path, **kwargs)
    (Path.home() / '.pipetorchuser').mkdir(exist_ok=True)
    df.to_csv(Path.home() / '.pipetorchuser' / filename, index=False)
    return df

def read_csv(path, filename=None, alternativesource=None, sep=None, delimiter=None, **kwargs):
    return PTDataFrame(read_pd_csv(path, filename=None, alternativesource=None, sep=None, delimiter=None, **kwargs))

class show_warning:
    def __enter__(self):
        self.warning = warnings.catch_warnings(record=True)
        self.w = self.warning.__enter__()
        warnings.filterwarnings('error')
        warnings.simplefilter('default')
        return self
    
    def __exit__(self, exc_type, exc_value, exc_traceback):
        for wi in self.w:
            if wi.line is None:
                print(wi.filename)
                wi.line = linecache.getline(wi.filename, wi.lineno)
        print(f'line number {wi.lineno}  line {wi.line}') 
        self.warning.__exit__(exc_type, exc_value, exc_traceback)

class PT:
    _metadata = ['_pt_scale_columns', '_pt_scale_omit_interval', '_pt_scalertype', '_pt_columny', '_pt_columnx', '_pt_transposey', '_pt_bias', '_pt_polynomials', '_dtype', '_pt_category', '_pt_category_sort', '_pt_sequence_window', '_pt_sequence_shift_y', '_pt_shuffle', '_pt_split', '_pt_random_state', '_pt_balance', '_pt_valid_dataframe', '_pt_test_dataframe']

    _internal_names = pd.DataFrame._internal_names + ['_pt__scale_columns', '_pt__train', '_pt__valid', '_pt__test', '_pt__full', '_pt__scalerx', '_pt__scalery', '_pt__train_indices', '_pt__train_x', '_pt__train_y', '_pt__valid_x', '_pt__valid_y', '_pt__categoryx', '_pt__categoryy', '_pt__indices', '_pt_train__indices', '_pt__valid_indices', '_pt__test_indices']
    _internal_names_set = set(_internal_names)

    @classmethod
    def read_csv(cls, path, **kwargs):
        df = pd.read_csv(path, **kwargs)
        return cls(df)

    @classmethod
    def from_dfs(cls, train, valid=None, test=None, **kwargs):
        df = cls(train)
        if valid is not None:
            df._pt_valid_dataframe = valid
        if test is not None:
            df._pt_test_dataframe = test
        return df
    
    def __init__(self, data, **kwargs):
        #super().__init__(data, **kwargs)
        if type(data) != self._constructor:
            for m in self._metadata:
                self.__setattr__(m, None)
                try:
                    self.__setattr__(m, getattr(data, m))
                except: pass

#     def groupby(self, by, axis=0, level=None, as_index=True, sort=True, group_keys=True, observed=False, dropna=True):
#         r = super().groupby(by, axis=axis, level=level, as_index=as_index, sort=sort, group_keys=group_keys, observed=observed, dropna=dropna)
#         return PTGroupedDataFrame(r)

    @property
    def _columny(self):
        try:
            if len(self._pt_columny) > 0:
                return self._pt_columny
        except: pass
        return [ self.columns[-1] ]
        
    @property
    def _transposey(self):
        return self._pt_transposey
        
    def astype(self, dtype, copy=True, errors='raise'):
        self._dtype = dtype
        return super().astype(dtype, copy=copy, errors=errors)
        
    @property
    def _columnx(self):
        if self._pt_columnx is None:
            return [ c for c in self.columns if c not in self._columny ]
        return self._pt_columnx
   
    @property
    def _columnsx_scale_indices(self):
        if self._pt_polynomials is not None:
            X = self.train._x_polynomials
            return [ i for i in range(X.shape[1]) if (X[:,i].min() < self._pt_scale_omit_interval[0] or X[:,i].max() > self._pt_scale_omit_interval[1]) ]
        columnx = self._columnx
        cat = set(self._pt_category) if type(self._pt_category) == tuple else []
        if self._pt_scale_columns == True or self._pt_scale_columns == 'x_only':
            r = [ c for c in columnx if c not in cat ]
        elif self._pt_scale_columns == False or self._pt_scale_columns is None or len(self._pt_scale_columns) == 0:
            r = []
        else:
            r = [ c for c in columnx if c in self._pt_scale_columns and c not in cat ]
        return [ columnx.index(c) for c in r ]

    @property
    def _columnsy_scale_indices(self):
        columny = self._columny
        cat = set(self._pt_category) if type(self._pt_category) == tuple else []
        if self._pt_scale_columns == True:
            y = self.train._y_numpy
            r = [ c for i, c in enumerate(columny) if c not in cat and (y[:,i].min() < self._pt_scale_omit_interval[0] or y[:,i].max() > self._pt_scale_omit_interval[1]) ]
        elif self._pt_scale_columns == False or self._pt_scale_columns is None or len(self._pt_scale_columns) == 0:
            r = []
        else:
            r = [ c for c in columny if c in self._pt_scale_columns and c not in cat ]
        return [ columny.index(c) for c in r ]

    def _scalerx(self):
        try:
            if self._pt__scalerx is not None:
                return self._pt__scalerx
        except: pass
        X = self.train._x_polynomials
        self._pt__scalerx = [ None ] * X.shape[1]
        for i in self._columnsx_scale_indices:
            self._pt__scalerx[i] = self._create_scaler(self._pt_scalertype, X[:, i:i+1])
        return self._pt__scalerx
        
    def _scalery(self):
        try:
            if self._pt__scalery is not None:
                return self._pt__scalery
        except: pass
        y = self.train._y_numpy
        self._pt__scalery = [ None ] * y.shape[1]
        for i in self._columnsy_scale_indices:
            self._pt__scalery[i] = self._create_scaler(self._pt_scalertype, y[:, i:i+1])
        return self._pt__scalery
    
    @property
    def _categoryx(self):
        try:
            if self._pt_category is None or len(self._pt_category) == 0:
                return None
            if self._pt__categoryx is not None:
                return self._pt__categoryx
        except: pass
        self._pt__categoryx = [ self._create_category(c) for c in self._columnx ]
        return self._pt__categoryx            
    
    @property
    def _categoryy(self):
        try:
            if self._pt_category is None or len(self._pt_category) == 0:
                return None
            if self._pt__categoryy is not None:
                return self._pt__categoryy
        except: pass
        self._pt__categoryy = [ self._create_category(c) for c in self._columny ]
        return self._pt__categoryy

    def columny(self, columns=None, transpose=None):
        """
        By default, PipeTorch uses the last column as the target variable and transposes it to become a row vector.
        This function can alter this default behavior. Transposing y is the default for single variable targets, 
        since most loss functions and metrics cannot handle column vectors. The set target variables are 
        automatically excluded from the input X.
        columns: single column name or list of columns that is to be used as target column. 
        None means use the last column
        transpose: True/False whether to transpose y. This is the default for single variable targets, since
        most loss functions and metrics expect a row vector. When a list of columns is used as a target
        transpose always has to be False.
        return: dataframe for which the given columns are marked as target columns, and marks whether the 
        target variable is to be transposed.
        """
        r = copy.deepcopy(self)
        if columns is not None:
            r._pt_columny = [columns] if type(columns) == str else columns
        if r._pt_columny is not None and len(r._pt_columny) > 1:
            transpose = False
        if transpose is not None:
            r._pt_transposey = transpose
        return r

    def columnx(self, *columns):
        r = copy.deepcopy(self)
        r._pt_columnx = list(columns) if len(columns) > 0 else None
        return r
    
    def category(self, *columns, sort=False):
        """
        Converts the values in the targetted columns into indices, for example to use in lookup tables.
        columns that are categorized are excluded from scaling. You cannot use this function together
        with polynomials or bias.
        columns: list of columns that is to be converted into a category
        sort: True/False (default False) whether the unique values of these colums should be converted to indices in sorted order.
        return: dataframe where the columns are converted into categories, 
        for which every unique value is converted into a unique index starting from 0
        """
        assert self._pt_polynomials is None, 'You cannot combine categories with polynomials'
        r = copy.copy(self)
        r._pt_category = columns
        r._pt_category_sort = sort
        return r
    
    def sequence(self, window, shift_y = 1):
        r = copy.copy(self)
        r._pt_sequence_window = window
        r._pt_sequence_shift_y = shift_y
        return r
    
    def _create_category(self, column):
        sort = self._pt_category_sort
        class Category:
            def fit(self, X):
                s = X.unique()
                if sort:
                    s = sorted(s)
                self.dict = { v:i for i, v in enumerate(s) }
                self.inverse_dict = { i:v for i, v in enumerate(s) }
            
            def transform(self, X):
                return X.map(self.dict)
            
            def inverse_transform(self, X):
                return X.map(self.inverse_dict)
            
        if column not in self._pt_category:
            return None
        
        c = Category()
        c.fit(self[column])
        return c
    
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
        return self._sequence_window+self._shift_y-1
    
    @property
    def _indices_unshuffled(self):
        if self._pt_sequence_window is not None:
            indices = list(range(len(self) - (self._pt_sequence_window + self._pt_sequence_shift_y - 1)))
            return indices
        else:
            return np.where(self.notnull().all(1))[0]

    @property
    def _indices(self):
        try:
            if self._pt__indices is not None:
                return self._pt__indices
        except: pass
        self._pt__indices = self._indices_unshuffled
        if (self._pt_shuffle is None or self._pt_shuffle) and self._pt_sequence_window is None:
            if self._pt_random_state is not None:
                np.random.seed(self._pt_random_state)
            np.random.shuffle(self._pt__indices)
            if self._pt_random_state is not None:
                t = 1000 * time.time() # current time in milliseconds
                np.random.seed(int(t) % 2**32)
        return self._pt__indices
        
    @property
    def _valid_begin(self):
        try:
            return int((1 - sum(self._pt_split))* len(self._indices))
        except: 
            try:
                return int((1 - self._pt_split)* len(self._indices))
            except:
                return len(self._indices)
        
    @property
    def _test_begin(self):
        try:
            return int((1 - self._pt_split[1])* len(self._indices))
        except:
            return len(self._indices)

    @property
    def _train_indices_unbalanced(self):
        return self._indices[:self._valid_begin]

    @property
    def _train_indices(self):
        try:
            return self._pt__train_indices
        except:
            indices = self._train_indices_unbalanced
            if self._pt_balance is not None:
                y = self.iloc[indices][self._columny]
                classes = np.unique(y)
                classindices = {c:np.where(y==c)[0] for c in classes}
                classlengths = {c:len(indices) for c, indices in classindices.items()}
                if self._pt_balance == True: # equal classes
                    n = max(classlengths.values())
                    mask = np.hstack([np.random.choice(classindices[c], n-classlengths[c], replace=True) for c in classes])
                else:                        # use given weights
                    n = max([ int(math.ceil(classlengths[c] / w)) for c, w in weights.items() ])
                    mask = np.hstack([np.random.choice(classindices[c], n*weights[c]-classlengths[c], replace=True) for c in classes])
                indices = np.array(indices)[ np.hstack([mask, list(range(len(y)))]) ]
            self._pt__train_indices = indices
            return self._pt__train_indices

    @property
    def _valid_indices(self):
        return self._indices[self._valid_begin:self._test_begin]

    @property
    def _test_indices(self):
        return self._indices[self._test_begin:]

    def to_dataset(self):
        """
        returns: a list with a train, valid and test DataSet. Every DataSet contains an X and y, where the 
        input data matrix X contains all columns but the last, and the target y contains the last column
        columns: list of columns to convert, the last column is always the target. default=None means all columns.
        """
        import torch
        from torch.utils.data import TensorDataset, DataLoader
        return [ self.train.to_dataset(), self.valid.to_dataset(), self.test.to_dataset() ]
    
    def to_databunch(self, batch_size=32, num_workers=0, shuffle=True, pin_memory=False, balance=False):
        """
        returns: a Databunch that contains dataloaders for the train, valid and test part.
        batch_size, num_workers, shuffle, pin_memory: see Databunch/Dataloader constructor
        """
        return Databunch(self, *self.to_dataset(), batch_size=batch_size, num_workers=num_workers, shuffle=shuffle, pin_memory=pin_memory, scaler=self, balance=balance)    

    def evaluate(self, *metrics):
        #assert len(metrics) > 0, 'You need to provide at least one metric for the evaluation'
        return Evaluator(self, *metrics)
   
    def _ptdataframe(self, data):
        return self._copy_meta( PTDataFrame(data) )

    def _ptdataset(self, data, indices=None):
        if indices is None:
            indices = list(range(len(data)))
        return PTDataSet.from_ptdataframe(data, self, indices)

    def _copy_meta(self, r):
        r._pt_scale_columns = self._pt_scale_columns
        r._pt_scale_omit_interval = self._pt_scale_omit_interval
        r._pt_scalertype = self._pt_scalertype
        r._pt_category = self._pt_category
        r._pt_category_sort = self._pt_category_sort
        r._pt_columny = self._pt_columny
        r._pt_columnx = self._pt_columnx
        r._pt_transposey = self._pt_transposey
        r._pt_polynomials = self._pt_polynomials
        r._pt_split = self._pt_split
        r._pt_random_state = self._pt_random_state
        r._pt_shuffle = self._pt_shuffle
        r._pt_balance = self._pt_balance
        r._pt_bias = self._pt_bias
        r._dtype = self._dtype
        r._pt_sequence_window = self._pt_sequence_window
        r._pt_sequence_shift_y = self._pt_sequence_shift_y
        r._pt_balance = self._pt_balance
        return r
    
    def split(self, split=0.2, shuffle=True, random_state=None):
        assert self._pt_valid_dataframe is None, 'You cannot combine split() with a fixed validation set'
        try:
            split[1]
            assert self._pt_test_dataframe is None, 'You cannot combine split(n,m) with a fixed validation set'
        except: pass
        r = copy.copy(self)
        r._pt_split = split
        r._pt_shuffle = shuffle
        r._pt_random_state = random_state
        return r
        
    def polynomials(self, degree):
        #assert not self._check_attr('_bias'), 'You should not add a bias before polynomials, rather use include_bias=True'
        assert type(self._pt_scale_columns) != list or len(self._pt_scale_columns) == 0, 'You cannot combine polynomials with column specific scaling'
        r = copy.copy(self)
        r._pt_polynomials = PolynomialFeatures(degree, include_bias=False)
        return r
    
    def from_numpy(self, x):
        if x.shape[1] == len(self._columnx) + len(self._columny):
            y = x[:,-len(self._columny):]
            x = x[:,:-len(self._columny)]
        elif x.shape[1] == len(self._columnx):
            y = np.zeros((len(x), len(self._columny)))
        else:
            raise ValueError('x must either have as many columns in x or the entire df')
        series = [ pd.Series(s.reshape(-1), name=c) for s, c in zip(x.T, self._columnx)]
        series.extend([ pd.Series(s.reshape(-1), name=c) for s, c in zip(y.T, self._columny) ] )
        df = pd.concat(series, axis=1)
        return self._ptdataset(df)
    
    def from_list(self, x):
        return self.from_numpy(np.array(x))
    
    def add_bias(self):
        r = copy.copy(self)
        r._pt_bias = True
        return r
    
    def _ptdataset_indices(self, indices):
        if self._pt_sequence_window is None:
            return self._ptdataset(self.iloc[indices], indices)
        else:
            try:
                low, high = min(indices), max(indices) + self._sequence_window + self._shift_y - 1
                return self._ptdataset(self.iloc[low:high], list(range(low, high)))
            except:
                return self._ptdataset(self.iloc[:0], [])
    
    @property
    def full(self):
        try:
            return self._pt__full
        except:
            self._pt__full = self._ptdataset_indices(np.concatenate([self._train_indices, self._valid_indices]))
        return self._pt__full

    @property
    def train(self):
        try:
            return self._pt__train
        except:
            self._pt__train = self._ptdataset_indices(self._train_indices)
        return self._pt__train
    
    @property
    def valid(self):
        try:
            return self._pt__valid
        except:
            if self._pt_valid_dataframe is None:
                self._pt__valid = self._ptdataset_indices(self._valid_indices)
            else:
                self._pt__valid = self._ptdataset(self._pt_valid_dataframe, range(len(self._pt_valid_dataframe)))
        return self._pt__valid

    @property
    def test(self):
        try:
            return self._pt__test
        except:
            if self._pt_test_dataframe is None:
                self._pt__test = self._ptdataset_indices(self._test_indices)
            else:
                self._pt__test = self._ptdataset(self._pt_test_dataframe, range(len(self._pt_test_dataframe)))    
        return self._pt__test
    
    @property
    def train_X(self):
        try:
            return self._pt__train_x
        except:
            self._pt__train_x = self.train.X
        return self._pt__train_x
            
    @property
    def train_y(self):
        try:
            return self._pt__train_y
        except:
            self._pt__train_y = self.train.y
        return self._pt__train_y

    @property
    def valid_X(self):
        try:
            return self._pt__valid_x
        except:
            self._pt__valid_x = self.valid.X
        return self._pt__valid_x
       
    @property
    def valid_y(self):
        try:
            return self._pt__valid_y
        except:
            self._pt__valid_y = self.valid.y
        return self._pt__valid_y

    @property
    def test_X(self):
        return self.test.X
            
    @property
    def test_y(self):
        return self.test.y
    
    @property
    def full_X(self):
        return self.full.X
            
    @property
    def full_y(self):
        return self.full.y
    
    def scale(self, columns=True, scalertype=StandardScaler, omit_interval=(-2,2)):
        if self._pt_polynomials and columns != 'x_only':
            assert type(columns) != list or len(columns) == 0, 'You cannot combine polynomials with column specific scaling'
        r = copy.copy(self)
        r._pt_scale_columns = columns
        r._pt_scalertype = scalertype
        r._pt_scale_omit_interval = omit_interval
        return r

    def scalex(self, scalertype=StandardScaler, omit_interval=(-2,2)):
        return self.scale(columns='x_only', scalertype=scalertype, omit_interval=omit_interval)
    
    def loss_surface(self, model, loss, **kwargs):
        self.evaluate(loss).loss_surface(model, loss, **kwargs)
    
    @staticmethod
    def _create_scaler(scalertype, column):
        scaler = scalertype()
        scaler.fit(column)
        return scaler

    def inverse_transform_y(self, y):
        y = to_numpy(y)
        if len(y.shape) == 1:
            y = y.reshape(-1,1)
        df = pd.DataFrame(self._inverse_transform(to_numpy(y), self._scalery(), self._columny))
        if self._categoryy is not None:
            for c, cat in zip(self._columny, self._categoryy):
                if cat is not None:
                    df[c] = cat.inverse_transform(df[c])
        return df

    def add_column(self, y, indices, erase_y=True, columns=None):
        df_y = self.inverse_transform_y(y)
        r = copy.deepcopy(self)
        if columns is None:
            columns = [ c + '_pred' for c in self._columny ]
        r[columns] = np.NaN
        r.loc[r.index[indices], columns] = df_y.values
        return self._ptdataframe(r)
    
    def inverse_transform_X(self, X):
        if self._pt_bias:
            X = X[:, 1:]
        if self._pt_polynomials is not None:
            X = X[:, :len(self._columnx)]
        df = self._inverse_transform(to_numpy(X), self._scalerx()[:len(self._columnx)], self._columnx)
        if self._categoryx is not None:
            for c, cat in zip(self._columnx, self._categoryx):
                if cat is not None:
                    df[c] = cat.inverse_transform(df[c])
        return df

    def _inverse_transform(self, data, scalerlist, columns):
        data = to_numpy(data)
        if scalerlist is not None:
            data = [ data[:, i:i+1] if scaler is None else scaler.inverse_transform(data[:,i:i+1]) for i, scaler in enumerate(scalerlist) ]
        series = [ pd.Series(x.reshape(-1), name=c) for x, c in zip(data, columns)]
        return pd.concat(series, axis=1)

    def inverse_transform(self, X, y, y_pred = None, cum=None):
        y = self.inverse_transform_y(y)
        X = self.inverse_transform_X(X)
        if y_pred is not None:
            y_pred = self.inverse_transform_y(y_pred).add_prefix('pred_')
            df = pd.concat([X, y, y_pred], axis=1)
        else:
            df = pd.concat([X, y], axis=1)
        if cum is not None:
            df = pd.concat([cum, df], axis=0)
        df = self._ptdataset(df)
        return df
    
    def balance(self, weights=True):
        r = copy.copy(self)
        r._pt_balance = weights
        return r
        
    def plot_boundary(self, predict):
        self.evaluate().plot_boundary(predict)
        
class PTDataFrame(pd.DataFrame, PT):
    _metadata = PT._metadata
    _internal_names = PT._internal_names
    _internal_names_set = PT._internal_names_set

    def __init__(self, data, *args, **kwargs):
        super().__init__(data, *args, **kwargs)
        PT.__init__(self, data)
    
    @property
    def _constructor(self):
        return PTDataFrame
    
    def groupby(self, by, axis=0, level=None, as_index=True, sort=True, group_keys=True, observed=False, dropna=True):
        r = super().groupby(by, axis=axis, level=level, as_index=as_index, sort=sort, group_keys=group_keys, observed=observed, dropna=dropna)
        return self._copy_meta( PTGroupedDataFrame(r) )

class PTSeries(pd.Series, PT):
    _metadata = PT._metadata
    _internal_names = PT._internal_names
    _internal_names_set = PT._internal_names_set

    def __init__(self, data, *args, **kwargs):
        super().__init__(data, *args, **kwargs)
        PT.__init__(self, data)

    @property
    def _constructor(self):
        return PTSeries
    
    @property
    def _constructor_expanddim(self):
        return PTDataFrame
    
class PTGroupedSeries(SeriesGroupBy, PT):
    _metadata = PT._metadata
    _internal_names = PT._internal_names
    _internal_names_set = PT._internal_names_set

    def __init__(self, data, *args, **kwargs):
        super().__init__(data, *args, **kwargs)
        PT.__init__(self, data)

    @property
    def _constructor(self):
        return PTGroupedSeries
    
    @property
    def _constructor_expanddim(self):
        return PTGroupedDataFrame
    
    
class PTGroupedDataFrame(DataFrameGroupBy, PT):
    _metadata = PT._metadata
    _internal_names = PT._internal_names
    _internal_names_set = PT._internal_names_set

    def __init__(self, data=None):
        super().__init__(obj=data.obj, keys=data.keys, axis=data.axis, level=data.level, grouper=data.grouper, exclusions=data.exclusions,
                selection=data._selection, as_index=data.as_index, sort=data.sort, group_keys=data.group_keys,
                observed=data.observed, mutated=data.mutated, dropna=data.dropna)
        PT.__init__(self, data)

    @property
    def _constructor(self):
        return PTGroupedDataFrame
    
    @property
    def _constructor_sliced(self):
        return PTGroupedSeries
    
    def astype(self, dtype, copy=True, errors='raise'):
        PTDataFrame.astype(self, dtype, copy=copy, errors=errors)

    def get_group(self, name, obj=None):
        return self._ptdataframe( super().get_group(name, obj=obj) )
        
    def to_dataset(self):
        from torch.utils.data import ConcatDataset
        dss = []
        for key, group in self:
            dss.append( self._ptdataframe(group).to_dataset())

        return [ConcatDataset(ds) for ds in zip(*dss)]
