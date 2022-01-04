import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, PolynomialFeatures, OneHotEncoder
from sklearn.utils import resample
from sklearn.model_selection import KFold
import matplotlib.pyplot as plt
import copy
import math
import time
import warnings
import linecache
from ..evaluate.evaluate import Evaluator
from .databunch import Databunch
from .dset import DSet
from pandas.core.groupby.generic import DataFrameGroupBy, SeriesGroupBy
from collections import defaultdict

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

class _DFrame:
    _metadata = ['_pt_scale_columns', '_pt_scale_omit_interval', '_pt_scalertype', '_pt_columny', '_pt_columnx', 
                 '_pt_transposey', '_pt_bias', '_pt_polynomials', '_pt_dtype', '_pt_category', '_pt_category_sort', 
                 '_pt_dummies', '_pt_sequence_window', '_pt_sequence_shift_y', 
                 '_pt_shuffle', '_pt_split', '_pt_random_state', '_pt_balance', 
                 '_pt_len', '_pt_indices', '_pt_train_valid_indices', '_pt_test_indices',
                 '_pt_dataset', '_pt_transforms', '_pt_train_transforms']

    _locked_names = ['_pt__locked_indices', '_pt__locked_train_indices', 
                       '_pt__locked_valid_indices', '_pt__locked_test_indices',
                       '_pt__locked_train', '_pt__locked_valid', 
                       '_pt__locked_scalerx', '_pt__locked_scalery',
                       '_pt__locked_categoryx', '_pt__locked_categoryy', 
                       '_pt__locked_dummiesx', '_pt__locked_dummiesy' ]

    _internal_names = pd.DataFrame._internal_names + _locked_names
    
    _internal_names_set = set( _internal_names )
    
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
            self._pt_transposey = False
            try:
                self.__setattr__(m, getattr(data, m))
            except: pass

    def _copy_meta(self, r):
        for c in self._metadata:
            setattr(r, c, getattr(self, c))
        return r
            
    def _copy_indices(self, r):
        if self.is_locked:
            r._pt__locked_indices = self._pt__locked_indices
            r._pt__locked_train_indices = self._pt__locked_train_indices
            r._pt__locked_valid_indices = self._pt__locked_valid_indices
            r._pt__locked_test_indices = self._pt__locked_test_indices
        return r
        
    def _dframe(self, data):
        return self._copy_meta( DFrame(data) )
    
    def _copy_with_indices(self):
        r = copy.copy(self)
        self._copy_indices(r)
        return r
        
    def _dset(self, data, indices=None, transforms=None):
        if indices is None:
            indices = list(range(len(data)))
        return DSet.from_dframe(data, self, indices, transforms)
    
    @property
    def is_locked(self):
        try:
            return self._pt__locked_indices is not None
        except:
            return False
    
    def lock(self):
        """
        To provide stable sampling of the train, valid and test set, this locks the sampled indices so that
        from this point, train, valid and test consistently produce the same subsets.
        
        Returns: DFrame
            a copy of the DFrame for which the sampled indices are locked
        """
        if not self.is_locked:
            self._pt__locked_indices = self._indices
            self._pt__locked_train_indices = self._train_indices
            self._pt__locked_valid_indices = self._valid_indices
            self._pt__locked_test_indices = self._test_indices
    
    @property
    def _columny(self):
        try:
            if len(self._pt_columny) > 0:
                return self._pt_columny
        except: pass
        return [ self.columns[-1] ]
        
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

    @staticmethod
    def _create_scaler(scalertype, column):
        scaler = scalertype()
        scaler.fit(column)
        return scaler
    
    @property
    def _scalerx(self):
        try:
            return self._pt__locked_scalerx
        except:  
            X = self.train._x_polynomials
            s = [ None ] * X.shape[1]
            for i in self._columnsx_scale_indices:
                s[i] = self._create_scaler(self._pt_scalertype, X[:, i:i+1])
            self._pt__locked_scalerx = s
            return self._pt__locked_scalerx
        
    @property
    def _scalery(self):
        try:
            return self._pt__locked_scalery
        except:
            y = self.train._y_numpy
            s = [ None ] * y.shape[1]
            for i in self._columnsy_scale_indices:
                s[i] = self._create_scaler(self._pt_scalertype, y[:, i:i+1])
            self._pt__locked_scalery = s
            return self._pt__locked_scalery
    
    def _create_category(self, column):
        sort = self._pt_category_sort
        class Category:
            def fit(self, X):
                s = X.unique()
                if sort:
                    s = sorted(s)
                self.dict = defaultdict(lambda:0, { v:(i+1) for i, v in enumerate(s) })
                self.inverse_dict = { (i+1):v for i, v in enumerate(s) }
                self.inverse_dict[0] = np.NaN
            
            def transform(self, X):
                return X.map(self.dict)
            
            def inverse_transform(self, X):
                return X.map(self.inverse_dict)
            
        if column not in self._pt_category:
            return None
        
        c = Category()
        c.fit(self.train[column])
        return c
    
    def _categoryx(self):
        try:
            return self._pt__locked_categoryx
        except:
            assert self.is_locked, '_categoryx can only be called on a locked DFrame to avoid consistency issues'
            if self._pt_category is None or len(self._pt_category) == 0:
                self._pt__locked_categoryx = None
            else:
                self._pt__locked_categoryx = [ self._create_category(c) for c in self._columnx ] 
            return self._pt__locked_categoryx
    
    def _categoryy(self):
        try:
            return self._pt__locked_categoryy
        except:
            assert self.is_locked, '_categoryy can only be called on a locked DFrame to avoid consistency issues'
            if self._pt_category is None or len(self._pt_category) == 0:
                self._pt__locked_categoryy = None
            else:
                self._pt__locked_categoryy = [ self._create_category(c) for c in self._columny ] 
            return self._pt__locked_categoryy

    def _create_dummies(self, column):    
        if column not in self._pt_dummies:
            return None
        
        c = OneHotEncoder(handle_unknown='ignore')
        c.fit(self.train[[column]])
        return c
    
    def _dummiesx(self):
        try:
            return self._pt__locked_dummiesx
        except:
            assert self.is_locked, '_dummiesx can only be called on a locked DFrame to avoid consistency issues'
            if self._pt_dummies is None or len(self._pt_dummies) == 0:
                self._pt__locked_dummiesx = [ None ] * len(self._columnx)
            else:
                self._pt__locked_dummiesx = [ self._create_dummies(c) for c in self._columnx ]
            return self._pt__locked_dummiesx
    
    def _dummiesy(self):
        try:
            return self._pt__locked_dummiesy
        except:
            assert self.is_locked, '_dummiesy can only be called on a locked DFrame to avoid consistency issues'
            if self._pt_dummies is None or len(self._pt_dummies) == 0:
                self._pt__locked_dummiesy = [ None ] * len(self._columny)
            else:
                self._pt__locked_dummiesy = [ self._create_dummies(c) for c in self._columny ]
            return self._pt__locked_dummiesy

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
            if self._pt__locked_indices:
                return self._pt__locked_indices
        except: pass
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
        try:
            if self._pt__locked_train_indices:
                return self._pt__locked_train_indices
        except: pass
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
                mask = np.hstack([self._pseudo_choose(classindices[c], round(n*weights[c])) for c in classes])
            indices = np.array(indices)[ mask ]
        return indices

    @property
    def _valid_indices(self):
        try:
            if self._pt__locked_valid_indices:
                return self._pt__locked_valid_indices
        except: pass
        return self._indices[self._valid_begin:self._test_begin]

    @property
    def _test_indices(self):
        try:
            if self._pt__locked_test_indices:
                return self._pt__locked_test_indices
        except: pass
        try:
            if self._pt_test_indices is not None:
                return self._pt_test_indices
        except: pass
        return self._indices[self._test_begin:]

    def df_to_dataset(self, df):
        """
        Converts the given df to a DataSet using the pipeline of this DFrame.
        
        Arguments:
            df: DataFrame or DFrame
                to convert into a DataSet
        
        returns: Converts the given df to a DataSet.
        """
        assert self.is_locked, 'You can only use a locked DFrame, to prevent inconsistencies in the transformation'
        return r.df_to_dset(df).to_dataset()
        
    def to_datasets(self, dataset=None):
        """
        Locks the DFrame (for a consistent split) and 
        returns a list with a train, valid and (optionally) test DataSet. 
        
        Arguments:
            dataset: class (None)
                The DataSet class to use
        
        Returns: list(DataSet)
        """
        self.lock()
        self._pt_dataset = dataset
        res = [ self.train.to_dataset() ]
        if len(self._valid_indices) > 0:
            res.append(self.valid.to_dataset())
        if len(self._test_indices) > 0:
            res.append(self.test.to_dataset())
        return res
        
    def df_to_dset(self, df):
        """
        Converts a DataFrame to a DSet that has the same pipeline as this DFrame.
        
        Arguments:
            df: DataFrame
        
        Returns: DSet
        """
        assert self.is_locked, 'You can only use a locked DFrame, to prevent inconsistencies in the transformation'
        return self._dset(df, range(len(df)))
        
    def to_databunch(self, dataset=None, batch_size=32, num_workers=0, shuffle=True, pin_memory=False, balance=False):
        """
        returns: a Databunch that contains dataloaders for the train, valid and test part.
        batch_size, num_workers, shuffle, pin_memory: see Databunch/Dataloader constructor
        """
        return Databunch(self, *self.to_datasets(dataset=dataset), 
                         batch_size=batch_size, num_workers=num_workers, shuffle=shuffle, 
                         pin_memory=pin_memory, balance=balance)    

    def kfold(self, n_splits=5):
        """
        Prepare the PipeTorch DataFrame for k-fold cross validation.
        n_splits: the number of folds
        return: a sequence of k PipeTorch DataFrames across which every item is used in the validation set once
        and the training splits and validation splits are disjoint.
        """
        kf = KFold(n_splits=n_splits)
        for train_ind, valid_ind in kf.split(self._train_indices):
            r = copy.copy(self)
            r._train_indices = self._train_indices[train_ind]
            r._valid_indices = self._train_indices[valid_ind]
            yield r
    
    def evaluate(self, *metrics):
        self.lock()
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
        return self._dset(df)
    
    def from_list(self, x):
        return self.from_numpy(np.array(x))
    
    def _dset_indices(self, indices, transforms):
        if self._pt_sequence_window is None:
            return self._dset(self.iloc[indices], indices, transforms)
        else:
            try:
                low, high = min(indices), max(indices) + self._sequence_window + self._shift_y - 1
                return self._dset(self.iloc[low:high], list(range(low, high)), transforms)
            except:
                return self._dset(self.iloc[:0], [], transforms)
    
    @property
    def train(self):
        """
        Returns or creates a DSet and optionally trains transformation parameters, e.g. for image
        normalization or text tokenization. Transformations are only supported for PyTorch DataSets.
        
        Returns: DSet
            with the train subset of the DFrame
        """
        try:
            if self.is_locked:
                return self._pt__locked_train
        except: pass
        self.lock()
        self._pt__locked_train = self._dset_indices(self._train_indices, self._transforms())
        if self._train_transformation_parameters(self._pt__locked_train):
            self._pt__locked_train = self._dset_indices(self._train_indices, self._transforms())
        return self._pt__locked_train
    
    @property
    def raw_train(self):
        self.lock()
        return self._dset_indices(self._train_indices, self._transforms())
    
    @property
    def valid(self):
        try:
            if self.is_locked:
                return self._pt__locked_valid
        except: pass
        self.lock()
        self._pt__locked_valid = self._dset_indices(self._valid_indices, self._transforms(train=False))
        return self._pt__locked_valid

    @property
    def test(self):
        self.lock()
        if self._pt_sequence_window is None:
            return DSet.df_to_testset(self.iloc[self._test_indices], self, self._test_indices, self._transforms(train=False))
        else:
            low, high = min(self._test_indices), max(self._test_indices) + self._sequence_window + self._shift_y - 1
            return DSet.df_to_testset(self.iloc[low:high], self, list(range(low, high)), self._transforms(train=False))
    
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
    
    def loss_surface(self, model, loss, **kwargs):
        self.evaluate(loss).loss_surface(model, loss, **kwargs)
    
    def inverse_scale_y(self, y):
        """
        Inversely scale an output y vector.
        
        Arguments:
        y: Numpy array or PyTorch tensor
            with the output that where preprocessed by this DFrame or predicted by the model
        
        Return: Pandas DataFrame
            That is reconstructed from Numpy arrays/Pytorch tensors
            that are transformed back to the original scale. 
        """
        y = to_numpy(y)
        if len(y.shape) == 1:
            y = y.reshape(-1,1)
        df = pd.DataFrame(self._inverse_scale(to_numpy(y), self._scalery, self._columny))
        if self._categoryy() is not None:
            for c, cat in zip(self._columny, self._categoryy()):
                if cat is not None:
                    df[c] = cat.inverse_transform(df[c])
        return df

    def add_column(self, y, indices, erase_y=True, columns=None):
        """
        Adds a column with values for the target variable to the DataFrame. When applicable, the transformation
        for the target variable is automatically inverted. This is useful to evaluate or visualize
        results.
        
        Arguments:
            y: Numpy/PyTorch array
                values in the same for as those generated for the target variable 
                (possibly predictions) that will be added to the DataFrame
            erase_y: bool (True)
                whether the original target variable is removed 
        
        Returns: copy of DFrame 
            to which y is added 
        """
        df_y = self.inverse_scale_y(y)
        r = copy.deepcopy(self)
        if columns is None:
            columns = [ c + '_pred' for c in self._columny ]
        r[columns] = np.NaN
        r.loc[r.index[indices], columns] = df_y.values
        return self._dframe(r)
    
    def inverse_scale_X(self, X):
        """
        Inversely scale an input X matrix.
        
        Arguments:
        X: Numpy array or PyTorch tensor
            with the input features that where preprocessed by this DataFrame
        
        Return: Pandas DataFrame
            That is reconstructed from Numpy arrays/Pytorch tensors
            that are transformed back to the orignal scale. 
        """
        if self._pt_bias:
            X = X[:, 1:]
        if self._pt_polynomials is not None:
            X = X[:, :len(self._columnx)]
        df = self._inverse_scale(to_numpy(X), self._scalerx[:len(self._columnx)], self._columnx)
        if self._categoryx() is not None:
            for c, cat in zip(self._columnx, self._categoryx()):
                if cat is not None:
                    df[c] = cat.inverse_transform(df[c])
        return df

    def _inverse_scale(self, data, scalerlist, columns):
        data = to_numpy(data)
        if scalerlist is not None:
            data = [ data[:, i:i+1] if scaler is None else scaler.inverse_transform(data[:,i:i+1]) for i, scaler in enumerate(scalerlist) ]
        series = [ pd.Series(x.reshape(-1), name=c) for x, c in zip(data, columns)]
        return pd.concat(series, axis=1)

    def inverse_scale(self, X, y, y_pred = None, cum=None):
        """
        Reconstructs a DSet from Numpy arrays/Pytorch tensors
        that are scaled back to the original scale. 
        This is useful for evaluation or to visualize results.
        
        Arguments:
            X: Numpy array or PyTorch tensor 
                with the same format as input features that were generated by the DataFrame
            y: Numpy array or PyTorch tensor
                with the same format as the target variable that was generated by the
                DataFrame
            pred_y (optional): Numpy array or PyTorch tensor
                with the same format as the target variable that was generated by the
                DataFrame
            cum (optional): DSet
                an dataset to add the results to, 
                to accumulate predictions over several mini-batches.
        
        Returns: DSet 
        """
        y = self.inverse_scale_y(y)
        X = self.inverse_scale_X(X)
        if y_pred is not None:
            y_pred = self.inverse_scale_y(y_pred).add_suffix('_pred')
            df = pd.concat([X, y, y_pred], axis=1)
        else:
            df = pd.concat([X, y], axis=1)
        if cum is not None:
            df = pd.concat([cum, df])
        df = self._dset(df)
        return df
        
    def plot_boundary(self, predict):
        self.evaluate().plot_boundary(predict)

    def balance(self, weights=True):
        """
        Oversamples rows in the training set, so that the values of the target variable 
        are better balanced. Does not affect the valid/test set.
        
        Arguments:
            weights: True or dict
                when set to True, the target values of the training set are 
                uniformely distributed,
                otherwise a dictionary can be passed that map target values to the 
                desired fraction of the training set (e.g. {0:0.4, 1:0.6}).
        
        Returns: copy of DFrame
            schdules to balance the train set
        """
        r = self._copy_with_indices()
        r._pt_balance = weights
        return r    
  
    def scale(self, columns=True, scalertype=StandardScaler, omit_interval=(-2,2)):
        """
        Scales the features and target variable in the DataFrame.
        
        Arguments:
            columns: True, str or list of str (True)
                the columns to scale (True for all)
            scalertype: an SKLearn type scaler (StandardScaler)
            omit_interval: (-2,2) when colums is set to True
                all columns whose values lie outside the omit_interval,
        
        Return: copy of DFrame
            schedules scaling the indicated columns, 
            using a scaler that is fitted om the training set
            and applied to train, valid and test set.
        """
        if self._pt_polynomials and columns != 'x_only':
            assert type(columns) != list or len(columns) == 0, 'You cannot combine polynomials with column specific scaling'
        r = self._copy_with_indices()
        r._pt_scale_columns = columns
        r._pt_scalertype = scalertype
        r._pt_scale_omit_interval = omit_interval
        return r
    
    def scalex(self, scalertype=StandardScaler, omit_interval=(-2,2)):
        """
        Scale all input features.
        
        Arguments:
            scalertype: SKLearn scaler class (StandardScaler)
            omit_interval: (-2,2) features whose values lie within this interval are not scaled
        
        Returns: copy of DFrame 
            schedules all input features (with values outside omit_interval)
            to be scaled (see scale)
        """
        return self.scale(columns='x_only', scalertype=scalertype, omit_interval=omit_interval)
    
    def add_bias(self):
        """
        Adds a bias column in the pipeline.
        
        Returns: copy of DFrame
            schedules as bias column of 1's to be added to the input
        """
        r = self._copy_with_indices()
        r._pt_bias = True
        return r
    
    def split(self, split=0.2, shuffle=True, random_state=None):
        """
        Split the data in a train/valid/(test) set.
        The train and valid sets will be resampled.
        
        Arguments:
            split: the fraction that is used for the validation set (and optionally test set). 
                When a single digit is given (default 0.2), that fraction of rows in the DataFrame will be 
                used for the validation set and the remainder for the training set. When a tuple of 
                fractions is given, those fractions of rows will be assigned to a validation and test set.
            shuffle: shuffle the rows before splitting
            random_state: set a random_state for reproducible results
            
        Returns: copy of DFrame 
            schedules the rows to be split into a train, valid and (optionally) test set.
        """
        assert not self.is_locked, 'You cannot change the pipeline of a Locked DFrame'
        r = copy.copy(self)
        r._pt_split = split
        r._pt_shuffle = shuffle
        r._pt_random_state = random_state
        return r
        
    def polynomials(self, degree, include_bias=False):
        """
        Adds (higher-order) polynomials to the data pipeline.
        
        Arguments:
            degree: int - degree of the higher order polynomials (e.g. 2 for squared)
            include_bias: bool (False) - whether to generate a bias column
        
        returns: copy of DFrame 
            schedules to generate higher order polynomials over the input features.
        """
        assert type(self._pt_scale_columns) != list or len(self._pt_scale_columns) == 0, 'You cannot combine polynomials with column specific scaling'
        r = self._copy_with_indices()
        r._pt_polynomials = PolynomialFeatures(degree, include_bias=include_bias)
        return r
    
    def no_columny(self):
        """
        PipeTorch cannot currently handle a dataset without a target variable, 
        however, it can work by simply assigning one of the used input features
        as a target variable.
        
        Returns: copy of DFrame
        """
        r = self._copy_with_indices()
        r._pt_columny = [self._pt_columnx[0]]
        return r
    
    def columny(self, columns=None, transpose=False):
        """
        By default, PipeTorch uses the last column as the target variable and transposes it to become a row vector.
        This function can alter this default behavior. Transposing y is the default for single variable targets, 
        since most loss functions and metrics cannot handle column vectors. The set target variables are 
        automatically excluded from the input X.
        
        Arguments:
            columns: str or list of str
                single column name or list of columns that is to be used as target column. 
                None: use the last column
            transpose: bool (False)
                whether to transpose y. 
                When using a single target variable targets, setting this to True
                allows to transpose the generated target vector.
        
        Returns: DFrame 
            schedules the given column(s) to be used as target columns, 
            and marks whether the target variable is to be transposed.
        """
        
        r = self._copy_with_indices()
        if columns is not None:
            r._pt_columny = [columns] if type(columns) == str else columns
        assert r._pt_columny is None or len(r._pt_columny) == 1 or not transpose, 'You cannot transpose multiple target columns'
        r._pt_transposey = transpose
        return r

    def columnx(self, *columns, omit=False):
        """
        Specify which columns to use as input. The target variable is always excluded
        so if you want to add that (for sequence learning), you should copy the
        target column. 
        Unlocks the DFrame, train and valid sets are therefore resampled.
        
        Arguments:
            columns: str or list of str
                columns to be used as input features
            omit: bool (False)
                when True, all columns are used except the specified target column(s)
        
        Returns: copy of DFrame
            schedules the given columns to use as input.
        """
        r = self._copy_with_indices()
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
        Unlocks the DFrame, train and valid sets are therefore resampled.
        
        Note: PipeTorch only uses categories that are in the training set and uses category 0 as
        an unknown category number for categories in the validation and test set that are not known during training.
        This way, no future information is used. 
        
        Arguments:
            columns: str or list of str
                list of columns that is to be converted into a category
            sort: True/False (default False) 
                whether the unique values of these colums should be converted 
                to indices in sorted order.
        
        Returns: copy of the PipeTorch DataFrame
            the columns are scheduled for conversion into categories, 
            by converting every unique value into a unique index starting from 0
        
        """
        assert self._pt_polynomials is None, 'You cannot combine categories with polynomials'
        assert self._pt_bias is None, 'You cannot combine categories with polynomials'
        r = self._copy_with_indices()
        r._pt_category = columns
        r._pt_category_sort = sort
        return r
    
    def dummies(self, *columns):
        """
        Converts the values in the targetted columns into dummy variables.
        columns that are categorized are excluded from scaling. You cannot use this function together
        with polynomials or bias.
        Unlocks the DFrame, train and valid sets are therefore resampled.
        
        Note: PipeTorch only uses categories that are in the training set and uses category 0 as
        an unknown category number for categories in the validation and test set that are not known during training.
        This way, no future information is used. 
        
        Arguments:
            columns: str or list of str
                the columns that are to be converted into a category
            sort: bool (False)
                whether the unique values of these colums should be converted 
                to indices in sorted order.
        
        Returns: copy of DFrame 
            schedules to convert the columns into categories, 
            for which every unique value is converted into a unique index starting from 0
        
        """
        assert self._pt_polynomials is None, 'You cannot combine categories with polynomials'
        assert self._pt_bias is None, 'You cannot combine categories with polynomials'
        r = self._copy_with_indices()
        r._pt_dummies = columns
        return r

    def sequence(self, window, shift_y = 1):
        """
        The rows in the DataFrame are considered to be a continuous sequence over time. 
        From this sequence, input samples are created that contain the features over a 'window'
        of rows. Ths allows to learn a model to predict a target based on prior history. The
        samples are generated so that their window overlaps.
        Unlocks the DFrame, train and valid sets are therefore resampled.
        
        Samples with NaN's are automatically skipped. When DataFrames are grouped, samples will
        be created only within groups.
        
        Arguments:
        window: int
            the number of rows that is used a an input
        shift_y: int
            how many rows into the future the target variable is placed, 
            e.g. with a window=2 and shift_y=1, X0 would contain [x[0], x[1]] 
            while y0 would contain y[2], 
            the next sample X1 would be [x[1], x[2]] and y[3].
        
        Returns: copy of DFrame
            schedules generating samples of sequences, using a sliding window 
            for learning on sequences
        """
        r = self._copy_with_indices()
        r._pt_sequence_window = window
        r._pt_sequence_shift_y = shift_y
        return r
    
    def train_transforms(self, *transforms):
        """
        Configure a (list of) transformation function(s) that is called from the DataSet class to prepare the 
        train data.
        
        Arguments:
            *transforms: [ callable ]
                (list of) transformation function(s) that is called from the DataSet class to prepare the data.
        """
        r = self._copy_with_indices()
        r._pt_train_transforms = transforms
        return r
    
    def transforms(self, *transforms):
        """
        Configure a (list of) transformation function(s) that is called from the DataSet class to prepare the data.

        Arguments:
            *transforms: [ callable ]
                (list of) transformation function(s) that is called from the DataSet class to prepare the data.
        """
        r = self._copy_with_indices()
        r._pt_transforms = transforms
        return r

    def _train_transformation_parameters(self, train_dset):
        """
        Placeholder to extend DFrame to train transformations on the train DSet. This is used by
        ImageDFrame to normalize images and TextDFrame to tokenize text.
        Note that the mechanism assumes that the train DSet is generated first.
        
        Arguments: 
            train_dset: DSet
                The train DSet to learn the transformation parameters on
            
        Returns: bool (None)
            This function should return True when parameters were learned, to force DFrame to add a new
            list of transformations to the generated DSet.
        """
        pass
    
    def _pre_transforms(self):
        """
        Placeholder for standard transformations that precede configured transformations.
        """
        return []
    
    def _post_transforms(self):
        """
        Placeholder for standard transformations that succede configured transformations.
        """
        return []
    
    def _transforms(self, pre=True, train=True, standard=True, post=True):
        """
        Arguments:
            pre: bool (True) - whether to include pre transformations
            train: bool (True) - whether to include train transformations
            standard: bool (True) - whether to include standard transformations
            post: bool (True) - whether to include post transformations
            
        Returns: [ callable ]
            A list of transformations that is applied to the generated train DataSet 
        """
        t = []
        try:
            if pre:
                t.extend(self._pre_transforms())
        except: pass
        try:
            if train:
                t.extend(self._pt_train_transforms)
        except: pass
        try:
            t.extend(self._pt_transforms)
        except: pass
        try:
            if post:
                t.extend( self._post_transforms() )
        except: pass
        return t
    
class DFrame(pd.DataFrame, _DFrame):
    _metadata = _DFrame._metadata
    _internal_names = _DFrame._internal_names
    _internal_names_set = _DFrame._internal_names_set

    def __init__(self, data, *args, **kwargs):
        super().__init__(data, *args, **kwargs)
        _DFrame.__init__(self, data)
        
    @property
    def _constructor(self):
        return DFrame
    
    def groupby(self, by, axis=0, level=None, as_index=True, sort=True, group_keys=True, observed=False, dropna=True):
        """
        Groups a DFrame just like a Pandas DataFrame. This is useful for 
        learning sequences, since samples are only created using a sliding window
        within groups.
        
        Arguments:
            by: single column or list of columns
            for the other arguments, see pandas.DataFrame.groupby 
        
        Returns: pipetorch.PTGroupedDFrame
        """
        r = super().groupby(by, axis=axis, level=level, as_index=as_index, sort=sort, group_keys=group_keys, observed=observed, dropna=dropna)
        return self._copy_meta( GroupedDFrame(r) )

     
class DSeries(pd.Series, _DFrame):
    _metadata = _DFrame._metadata
    _internal_names = _DFrame._internal_names
    _internal_names_set = _DFrame._internal_names_set

    def __init__(self, data, *args, **kwargs):
        super().__init__(data, *args, **kwargs)
        _DFrame.__init__(self, data)

    @property
    def _constructor(self):
        return DSeries
    
    @property
    def _constructor_expanddim(self):
        return DFrame
    
class GroupedDSeries(SeriesGroupBy, _DFrame):
    _metadata = _DFrame._metadata
    _internal_names = _DFrame._internal_names
    _internal_names_set = _DFrame._internal_names_set

    def __init__(self, data, *args, **kwargs):
        super().__init__(data, *args, **kwargs)
        _DFrame.__init__(self, data)

    @property
    def _constructor(self):
        return GroupedDSeries
    
    @property
    def _constructor_expanddim(self):
        return GroupedDFrame
    
class GroupedDFrame(DataFrameGroupBy, _DFrame):
    _metadata = _DFrame._metadata
    _internal_names = _DFrame._internal_names
    _internal_names_set = _DFrame._internal_names_set

    def __init__(self, data=None):
        super().__init__(obj=data.obj, keys=data.keys, axis=data.axis, level=data.level, grouper=data.grouper, exclusions=data.exclusions,
                selection=data._selection, as_index=data.as_index, sort=data.sort, group_keys=data.group_keys,
                observed=data.observed, mutated=data.mutated, dropna=data.dropna)
        _DFrame.__init__(self, data)

    @property
    def _constructor(self):
        return GroupedDFrame
    
    @property
    def _constructor_sliced(self):
        return GroupedDSeries
    
    def astype(self, dtype, copy=True, errors='raise'):
        """
        see: pipetorch.DFrame.astype
        """
        DFrame.astype(self, dtype, copy=copy, errors=errors)

    def get_group(self, namel, obj=None):
        return self._dframe( super().get_group(name, obj=obj) )
        
    def __iter__(self):
        for group, sub0set in super().__iter__():
            yield group, self._copy_meta(subset)
        
    def to_dataset(self):
        """
        Convert 0a grouped DFrame into a PyTorch ConcatDataset over datasets
        for every group contained.
        
        returns: train, valid, test DataSet
        """
        from torch.utils.data import ConcatDataset
        dss = []
        for key, group in self:
            dss.append( self._dframe(group).to_dataset())

        return [ConcatDataset(ds) for ds in zip(*dss)]
