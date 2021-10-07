import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.utils import resample
import matplotlib.pyplot as plt
import copy
import time
import warnings
import linecache
from ..evaluate.evaluate import Evaluator
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
    _metadata = ['_pt_scale_columns', '_pt_scale_omit_interval', '_pt_scalertype', '_pt_columny', '_pt_columnx', '_pt_transposey', '_pt_bias', '_pt_polynomials', '_pt_dtype', '_pt_category', '_pt_category_sort', '_pt_sequence_window', '_pt_sequence_shift_y', '_pt_shuffle', '_pt_split', '_pt_random_state', '_pt_balance', '_pt_len', '_pt_indices', '_pt_train_valid_indices', '_pt_test_indices']

    @classmethod
    def read_csv(cls, path, **kwargs):
        df = pd.read_csv(path, **kwargs)
        return cls(df)

    @classmethod
    def from_dfs(cls, *dfs, **kwargs):
        return cls(pd.concat(dfs), **kwargs)
    
    @classmethod
    def from_train_test(cls, train, test, **kwargs):
        r = cls(pd.concat([train, test], ignore_index=True))
        r._pt_train_valid_indices = list(range(len(train)))
        r._pt_test_indices = list(range(len(train), len(train)+len(test)))
        return r
    
    def __init__(self, data, **kwargs):
        for m in self._metadata:
            self.__setattr__(m, None)
            try:
                self.__setattr__(m, getattr(data, m))
            except: pass

    def _copy_meta(self, r):
        for c in self._metadata:
            setattr(r, c, getattr(self, c))
        return r
                    
    def _ptdataframe(self, data):
        return self._copy_meta( PTDataFrame(data) )

    def _ptlockeddataframe(self, data):
        return self._copy_meta( PTLockedDataFrame(data) )

    def locked(self):
        return self._ptlockeddataframe(self)
    
    def _ptdataset(self, data, indices=None):
        if indices is None:
            indices = list(range(len(data)))
        return PTDataSet.from_ptdataframe(data, self, indices)
    
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
        self._pt_dtype = dtype
        return super().astype(dtype, copy=copy, errors=errors)
        
    @property
    def _columnx(self):
        if self._pt_columnx is None:
            return [ c for c in self.columns if c not in self._columny ]
        return [ c for c in self._pt_columnx if c not in self._columny ]
   
    @property
    def _all_columns(self):
        return list(set(self._columnx).union(set(self._columny)))

    @property
    def _columnsx_scale_indices(self):
        if self._pt_polynomials is not None and self._pt_scale_columns is not None:
            X = self.train._x_polynomials
            return [ i for i in range(X.shape[1]) if (X[:,i].min() < self._pt_scale_omit_interval[0] or X[:,i].max() > self._pt_scale_omit_interval[1]) ]
        columnx = self._columnx
        cat = set(self._pt_category) if type(self._pt_category) == tuple else []
        if self._pt_scale_columns == True or self._pt_scale_columns == 'x_only':
            r = [ c for c in columnx if c not in cat]
        elif self._pt_scale_columns == False or self._pt_scale_columns is None or len(self._pt_scale_columns) == 0:
            r = []
        else:
            r = [ c for c in columnx if c in self._pt_scale_columns and c not in cat ]
        X = self.train._x_polynomials
        r = [ columnx.index(c) for i, c in enumerate(columnx) if c in r and ((X[:,i].min() < self._pt_scale_omit_interval[0] or X[:,i].max() > self._pt_scale_omit_interval[1])) ]
        return r
        
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

    @property
    def _scalerx(self):
        X = self.train._x_polynomials
        s = [ None ] * X.shape[1]
        for i in self._columnsx_scale_indices:
            s[i] = self._create_scaler(self._pt_scalertype, X[:, i:i+1])
        return s
        
    @property
    def _scalery(self):
        y = self.train._y_numpy
        s = [ None ] * y.shape[1]
        for i in self._columnsy_scale_indices:
            s[i] = self._create_scaler(self._pt_scalertype, y[:, i:i+1])
        return s
    
    @property
    def _categoryx(self):
        try:
            if self._pt_category is None or len(self._pt_category) == 0:
                return None
        except: pass
        return [ self._create_category(c) for c in self._columnx ]          
    
    @property
    def _categoryy(self):
        try:
            if self._pt_category is None or len(self._pt_category) == 0:
                return None
        except: pass
        return [ self._create_category(c) for c in self._columny ]

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
            try:
                self._pt_train_valid_indices = [ i for i in self._pt_train_valid_indices if i in self.index ]
                indices = self._pt_train_valid_indices[:-(self._pt_sequence_window + self._pt_sequence_shift_y - 1)]
            except:
                indices = list(range(len(self) - (self._pt_sequence_window + self._pt_sequence_shift_y - 1)))
            return indices
        else:
            try:
                return [ i for i in self._pt_train_valid_indices if i in self.index ]
            except:
                return np.where(self[self._all_columns].notnull().all(1))[0]

    @property
    def _shuffle(self):
        return ((self._pt_shuffle is None and self._pt_split is not None) or self._pt_shuffle) and \
               self._pt_sequence_window is None
        
    def _check_len(self):
        """
        Internal method, to check if then length changed, to keep the split between the train/valid/test
        unless the length changed to obtain stable results
        """
        try:
            if self._pt_len == len(self._pt_indices):
                return True
        except: pass
        return False

    @property
    def _indices(self):
        try:
            if self._check_len():
                return self._pt_indices
        except: pass
        self._pt_indices = self._indices_unshuffled
        self._pt_len = len(self._pt_indices)
        if self._shuffle:
            if self._pt_random_state is not None:
                np.random.seed(self._pt_random_state)
            np.random.shuffle(self._pt_indices)
        return self._pt_indices
        
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

    def _pseudo_choose(self, indices, items):
        r = indices[:items % len(indices)]
        while items >= len(indices):
            r = np.hstack([r, indices])
            items -= len(indices)
        return r
    
    @property
    def _train_indices(self):
        indices = self._train_indices_unbalanced
        if self._pt_balance is not None:
            y = self.iloc[indices][self._columny]
            classes = np.unique(y)
            classindices = {c:np.where(y==c)[0] for c in classes}
            classlengths = {c:len(indices) for c, indices in classindices.items()}
            if self._pt_balance == True: # equal classes
                n = max(classlengths.values())
                mask = np.hstack([self._pseudo_choose(classindices[c], n) for c in classes])
            else:                        # use given weights
                weights = self._pt_balance
                n = max([ int(math.ceil(classlengths[c] / w)) for c, w in weights.items() ])
                mask = np.hstack([self._pseudo_choose(classindices[c], n*weights[c]) for c in classes])
            indices = np.array(indices)[ mask ]
        return indices

    @property
    def _valid_indices(self):
        return self._indices[self._valid_begin:self._test_begin]

    @property
    def _test_indices(self):
        try:
            if self._pt_test_indices is not None:
                return self._pt_test_indices
        except: pass
        return self._indices[self._test_begin:]

    def to_dataset(self, *dfs):
        """
        returns: a list with a train, valid and test DataSet. Every DataSet contains an X and y, where the 
        input data matrix X contains all columns but the last, and the target y contains the last column
        columns: list of columns to convert, the last column is always the target. default=None means all columns.
        """
        import torch
        from torch.utils.data import TensorDataset, DataLoader
        r = PTLockedDataFrame(self)
        if r._pt_transposey is None:
            r._pt_transposey = False
        res = [ r.train.to_dataset() ]
        if len(self._valid_indices) > 0:
            res.append(r.valid.to_dataset())
        if len(self._test_indices) > 0:
            res.append(r.test.to_dataset())
        for df in dfs:
            res.append(r.to_subset(df).to_dataset())
        assert len(res) < 4, 'You cannot have more than a train, valid and test set'
        return res
        
    def to_subset(self, df):
        return self._ptdataset(df, range(len(df)))
        
    def to_databunch(self, *dfs, batch_size=32, num_workers=0, shuffle=True, pin_memory=False, balance=False):
        """
        returns: a Databunch that contains dataloaders for the train, valid and test part.
        batch_size, num_workers, shuffle, pin_memory: see Databunch/Dataloader constructor
        """
        return Databunch(self, *self.to_dataset(*dfs), batch_size=batch_size, num_workers=num_workers, shuffle=shuffle, pin_memory=pin_memory, balance=balance)    

    def evaluate(self, *metrics):
        return Evaluator(self, *metrics)
   
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
        return self._ptdataset_indices(np.concatenate([self._train_indices, self._valid_indices]))

    @property
    def train(self):
        return self._ptdataset_indices(self._train_indices)
    
    @property
    def valid(self):
        return self._ptdataset_indices(self._valid_indices)

    @property
    def test(self):
        if self._pt_sequence_window is None:
            return PTDataSet.df_to_testset(self.iloc[self._test_indices], self, self._test_indices)
        else:
            low, high = min(self._test_indices), max(self._test_indices) + self._sequence_window + self._shift_y - 1
            return PTDataSet.df_to_testset(self.iloc[low:high], self, list(range(low, high)))
    
    @property
    def train_X(self):
        return self.train.X
            
    @property
    def train_y(self):
        return self.train.y

    @property
    def valid_X(self):
        return self.valid.X
       
    @property
    def valid_y(self):
        return self.valid.y

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
        df = pd.DataFrame(self._inverse_transform(to_numpy(y), self._scalery, self._columny))
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
        df = self._inverse_transform(to_numpy(X), self._scalerx[:len(self._columnx)], self._columnx)
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
            y_pred = self.inverse_transform_y(y_pred).add_suffix('_pred')
            df = pd.concat([X, y, y_pred], axis=1)
        else:
            df = pd.concat([X, y], axis=1)
        if cum is not None:
            df = pd.concat([cum, df])
        df = self._ptdataset(df)
        return df
        
    def plot_boundary(self, predict):
        self.evaluate().plot_boundary(predict)

class PTSet:
    def balance(self, weights=True):
        r = copy.copy(self)
        r._pt_balance = weights
        return r    
  
    def scale(self, columns=True, scalertype=StandardScaler, omit_interval=(-2,2)):
        if self._pt_polynomials and columns != 'x_only':
            assert type(columns) != list or len(columns) == 0, 'You cannot combine polynomials with column specific scaling'
        r = copy.copy(self)
        r._pt_scale_columns = columns
        r._pt_scalertype = scalertype
        r._pt_scale_omit_interval = omit_interval
        return r
    
    def scalex(self, scalertype=StandardScaler, omit_interval=(-2,2)):
        return self.scale(columns='x_only')
    
    def add_bias(self):
        r = copy.copy(self)
        r._pt_bias = True
        return r
    
    def split(self, split=0.2, shuffle=True, random_state=None):
        r = copy.copy(self)
        r._pt_split = split
        r._pt_shuffle = shuffle
        r._pt_random_state = random_state
        return r
        
    def polynomials(self, degree, include_bias=False):
        assert type(self._pt_scale_columns) != list or len(self._pt_scale_columns) == 0, 'You cannot combine polynomials with column specific scaling'
        r = copy.copy(self)
        r._pt_polynomials = PolynomialFeatures(degree, include_bias=include_bias)
        return r
    
    def no_columny(self):
        r = copy.deepcopy(self)
        self._pt_columny = [self._pt_columnx[0]]
        return r
    
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

    def columnx(self, *columns, omit=False):
        """
        Specify which columns to use as input. The target variable is always excluded
        so if you want to add that (for sequence learning), you should copy the
        target column. 
        omit: when True, all columns are used except the specified columns
        return: a new PipeTorch DataFrame with the given input specified.
        """
        r = copy.deepcopy(self)
        if omit:
            r._pt_columnx = [ c for c in self.columns if c not in columns ]
        else:
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
    
class PTDataFrame(pd.DataFrame, PT, PTSet):
    _metadata = PT._metadata

    def __init__(self, data, *args, **kwargs):
        super().__init__(data, *args, **kwargs)
        PT.__init__(self, data)
        
    @property
    def _constructor(self):
        return PTDataFrame
    
    def groupby(self, by, axis=0, level=None, as_index=True, sort=True, group_keys=True, observed=False, dropna=True):
        r = super().groupby(by, axis=axis, level=level, as_index=as_index, sort=sort, group_keys=group_keys, observed=observed, dropna=dropna)
        return self._copy_meta( PTGroupedDataFrame(r) )

class PTLockedDataFrame(pd.DataFrame, PT):
    _internal_names = ['_pt__scale_columns', '_pt__train', '_pt__valid', '_pt__test', '_pt__full', '_pt__scalerx', '_pt__scalery', '_pt__train_x', '_pt__train_y', '_pt__valid_x', '_pt__valid_y', '_pt__categoryx', '_pt__categoryy', '_pt__train_indices', '_pt__valid_indices', '_pt__test_indices']
    _metadata = PT._metadata + _internal_names

    def __init__(self, data, **kwargs):
        super().__init__(data, **kwargs)
        for m in self._metadata:
            try:
                self.__setattr__(m, getattr(data, m))
            except: pass

    @property
    def _constructor(self):
        return PTLockedDataFrame
    
    @property
    def _scalerx(self):
        try:
            if self._pt__scalerx is not None:
                return self._pt__scalerx
        except: pass
        self._pt__scalerx = super()._scalerx
        return self._pt__scalerx
        
    @property
    def _scalery(self):
        try:
            if self._pt__scalery is not None:
                return self._pt__scalery
        except: pass
        self._pt__scalery = super()._scalery
        return self._pt__scaler
    
    @property
    def _categoryx(self):
        try:
            if self._pt__categoryx is not None:
                return self._pt__categoryx
        except: pass
        self._pt__categoryx = super()._categoryx
        return self._pt__categoryx            
    
    @property
    def _categoryy(self):
        try:
            if self._pt__categoryy is not None:
                return self._pt__categoryy
        except: pass
        self._pt__categoryy = super()._categoryy
        return self._pt__categoryy            

    @property
    def full(self):
        try:
            return self._pt__full
        except:
            self._pt__full = super().full
            return self._pt__full

    @property
    def train(self):
        try:
            return self._pt__train
        except:
            self._pt__train = super().train
            return self._pt__train
    
    @property
    def valid(self):
        try:
            return self._pt__valid
        except:
            self._pt__valid = super().valid
            return self._pt__valid

    @property
    def test(self):
        try:
            return self._pt__test
        except:
            self._pt__test = super().test
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
        
class PTSeries(pd.Series, PT):
    _metadata = PT._metadata

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
        
    def __iter__(self):
        for group, subset in super().__iter__():
            yield group, self._copy_meta(subset)
        
    def to_dataset(self):
        from torch.utils.data import ConcatDataset
        dss = []
        for key, group in self:
            dss.append( self._ptdataframe(group).to_dataset())

        return [ConcatDataset(ds) for ds in zip(*dss)]

