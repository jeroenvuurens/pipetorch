import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, PolynomialFeatures, OneHotEncoder, FunctionTransformer
from sklearn.utils import resample
from sklearn.model_selection import KFold, train_test_split, StratifiedKFold, StratifiedShuffleSplit
from iterstrat.ml_stratifiers import MultilabelStratifiedKFold, MultilabelStratifiedShuffleSplit
import matplotlib.pyplot as plt
import copy
import math
import time
import random
import warnings
import linecache
from ..evaluate.evaluate import Evaluator
from .helper import read_from_package, read_from_function
from .kagglereader import Kaggle
from .databunch import Databunch
from .dset import DSet
from pandas.core.groupby.generic import DataFrameGroupBy, SeriesGroupBy
from collections import defaultdict
from contextlib import contextmanager
import functools
from pandas.util import hash_pandas_object
import hashlib
from pandas.util._exceptions import find_stack_level
import warnings

pd.options.mode.chained_assignment = None

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
    _config   = ['_pt_scale_columns', '_pt_scale_omit_interval', '_pt_scalertype', '_pt_columny', '_pt_columnx', 
                 '_pt_vectory', '_pt_bias', '_pt_polynomials', '_pt_dtype', '_pt_category', '_pt_category_sort', 
                 '_pt_dummies', '_pt_evaluator', '_pt_transform',
                 '_pt_dataset_transforms', '_pt_dataset_train_transforms' ]
    
    _config_indices = [
                 '_pt_sequence_window', '_pt_sequence_shift_y', 
                 '_pt_split_shuffle', '_pt_split_stratify', '_pt_split_stratify_test', 
                 '_pt_folds_stratify', '_pt_folds_random_state', 
                 '_pt_split_random_state', '_pt_folds_shuffle', 
                 '_pt_valid_size', '_pt_test_size', '_pt_balance', '_pt_filterna', 
                 '_pt_train_valid_indices', '_pt_test_indices', '_pt_folds', '_pt_fold',
                 '_cached_index' ]
    
    _config_fold = [ '_pt_fold' ]
    
    _cached =  [ #'_cached_changed', #'_cached_changed_fold', 
                 '_cached_train', '_cached_valid', '_cached_test', '_cached_raw_train',
                 '_cached_scalerx', '_cached_scalery',
                 '_cached_categoryx', '_cached_categoryy',
                 '_cached_columntransformerx', '_cached_columntransformery',
                 '_cached_dummiesx', '_cached_dummiesy']
                
    _cached_indices = [ '_cached_train_indices', '_cached_valid_indices', '_cached_test_indices',
                        '_cached_indices_before_testsplit', '_cached_indices_after_testsplit' ]
    
    _cached_fold = [ '_cached_folds' ]

    _metadata = _config + _config_indices + _config_fold + _cached + _cached_indices + _cached_fold
    _config_set = set( _config )
    _config_indices_set = set( _config_indices )
    _config_fold_set = set( _config_fold )
    
    _internal_names = pd.DataFrame._internal_names + _metadata
    
    _internal_names_set = set( _internal_names )
    
    def __init__(self, data, **kwargs):
        for m in self._config:
            setattr(self, m, None)
        for m in self._config_indices:
            setattr(self, m, None)
        self._index_changed()
        for m in self._metadata:
            try:
                setattr(self, m, getattr(data, m))
            except: pass
    
    @classmethod
    def stop_warnings(cls):
        pd.options.mode.chained_assignment = None
    
    def _copy_meta(self, r):
        for c in self._metadata:
            try:
                setattr(r, c, getattr(self, c))
            except: pass
        return r

    def _dframe(self, data):
        return self._copy_meta( DFrame(data) )

    def _dset(self, data, transforms=None):
        return DSet.from_dframe(data, self, transforms)
        
    @property
    def _columny(self):
        try:
            if len(self._pt_columny) > 0:
                return self._pt_columny
        except: pass
        return [ self.columns[-1] ]
        
    def dtype(self, dtype):
        """
        Lazily executed change of dtype to the data.
        """
        r = self.copy(deep=False)
        r._change('_pt_dtype', dtype)
        return r
        
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
            X = self.raw_train._x_polynomials
            return [ i for i in range(X.shape[1]) if (X[:,i].min() < self._pt_scale_omit_interval[0] or X[:,i].max() > self._pt_scale_omit_interval[1]) ]
        columnx = self._columnx
        cat = set(self._pt_category) if type(self._pt_category) == tuple else []
        if self._pt_scale_columns == True or self._pt_scale_columns == 'x_only':
            r = [ c for c in columnx if c not in cat]
        elif self._pt_scale_columns == False or self._pt_scale_columns is None or len(self._pt_scale_columns) == 0:
            r = []
        else:
            r = [ c for c in columnx if c in self._pt_scale_columns and c not in cat ]
        if len(r) > 0:
            X = self.raw_train._x_polynomials
            r = [ columnx.index(c) for i, c in enumerate(columnx) if c in r and ((X[:,i].min() < self._pt_scale_omit_interval[0] or X[:,i].max() > self._pt_scale_omit_interval[1])) ]
        return r
        
    @property
    def _columnsy_scale_indices(self):
        columny = self._columny
        cat = set(self._pt_category) if type(self._pt_category) == tuple else []
        if self._pt_scale_columns == True:
            y = self.raw_train._y_numpy
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
        if self._unchanged() and self._cached_scalerx is not None:
            return self._cached_scalerx
        
        X = self.raw_train._x_polynomials
        r = [ None ] * X.shape[1]
        for i in self._columnsx_scale_indices:
            r[i] = self._create_scaler(self._pt_scalertype, X[:, i:i+1])
        self._change('_cached_scalerx', r)
        return r
     
    @_scalerx.setter
    def _scalerx(self, value):
        self._change('_cached_scalerx', value)
        
    @property
    def _scalery(self):
        if self._unchanged() and self._cached_scalery is not None:
            return self._cached_scalery
        
        y = self.raw_train._y_numpy
        r = [ None ] * y.shape[1]
        for i in self._columnsy_scale_indices:
            r[i] = self._create_scaler(self._pt_scalertype, y[:, i:i+1])
        self._change('_cached_scalery', r)
        return r
    
    @_scalery.setter
    def _scalery(self, value):
        self._change('_cached_scalery', value)
        
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
        c.fit(self.raw_train[column])
        return c
    
    def _categoryx(self):
        if self._unchanged() and self._cached_categoryx is not None:
            return self._cached_categoryx
        
        if self._pt_category is None or len(self._pt_category) == 0:
            return None
        r = [ self._create_category(c) for c in self._columnx ]
        self._change('_cached_categoryx', r)
        return self._cached_categoryx
    
    def _categoryy(self):
        if self._unchanged() and self._cached_categoryy is not None:
            return self._cached_categoryy
        
        if self._pt_category is None or len(self._pt_category) == 0:
            return None
        r = [ self._create_category(c) for c in self._columny ] 
        self._change('_cached_categoryy', r)
        return r

    def _columntransformerx(self):
        if self._unchanged() and self._cached_columntransformerx is not None:
            return self._cached_columntransformerx
        
        if self._pt_transform is None or len(self._pt_transform) == 0:
            return None
        r = []
        for c in self._columnx:
            try:
                t = self._pt_transform[c]
                t.fit(self.raw_train[c])
                r.append(t)
            except:
                r.append(None)
                
        self._change('_cached_columntransformerx', r)
        return self._cached_columntransformerx
        
    def _columntransformery(self):
        if self._unchanged() and self._cached_columntransformery is not None:
            return self._cached_columntransformery
        
        if self._pt_transform is None or len(self._pt_transform) == 0:
            return None
        r = []
        for c in self._columny:
            try:
                t = self._pt_transform[c]
                t.fit(self.raw_train[c])
                r.append(t)
            except:
                r.append(None)
                
        self._change('_cached_columntransformery', r)
        return self._cached_columntransformery
    
    def _create_dummies(self, column):    
        if column not in self._pt_dummies:
            return None
        
        c = OneHotEncoder(handle_unknown='ignore')
        c.fit(self.raw_train[[column]])
        return c
    
    def _dummiesx(self):
        if self._unchanged() and self._cached_dummiesx is not None:
            return self._cached_dummiesx
        
        if self._pt_dummies is None or len(self._pt_dummies) == 0:
            r = [ None ] * len(self._columnx)
        else:
            r = [ self._create_dummies(c) for c in self._columnx ]
        self._change('_cached_dummiesx', r)
        return r
    
    def _dummiesy(self):
        if self._unchanged() and self._cached_dummiesy is not None:
            return self._cached_dummiesy
        
        if self._pt_dummies is None or len(self._pt_dummies) == 0:
            r = [ None ] * len(self._columny)
        else:
            r = [ self._create_dummies(c) for c in self._columny ]
        self._change('_cached_dummiesy', r)
        return r
        
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
        if self._pt_train_valid_indices is not None:
            return np.intersect1d(self.index.values, self._pt_train_valid_indices)
        return self.index.values

    @property
    def _indices_before_testsplit(self):
        if self._unchanged() and self._cached_indices_before_testsplit is not None:
            return self._cached_indices_before_testsplit
        
        test_indices = set(self.fixed_test_indices)
        if len(test_indices) > 0:
            r = np.array([ i for i in self._indices_unshuffled if i not in test_indices ])
        else:
            r = self._indices_unshuffled
        self._change('_cached_indices_before_testsplit', r)
        return r
           
    @property
    def _indices_after_testsplit(self):
        if self._unchanged() and self._cached_indices_after_testsplit is not None:
            return self._cached_indices_after_testsplit
        
        test_indices = set(self._test_indices)
        r = np.array([ i for i in self._indices_before_testsplit if i not in test_indices ])
        self._change('_cached_indices_after_testsplit', r)
        return r

    @property
    def _test_size(self):
        return self._pt_test_size or 0
    
    @property
    def _valid_size(self):
        return self._pt_valid_size or 0
    
    @property
    def _shuffle(self):
        return ((self._pt_split_shuffle is None and \
                (self._test_size > 0 or self._valid_size > 0)) \
                or self._pt_split_shuffle) and self._pt_sequence_window is None
    
    def _stratifyable_columns(self, indices, columns, bins):
        if columns == True:
            columns = self._all_columns
        df = self.loc[indices, columns]
        if bins > 1:
            bins = len(df) // bins
        else:
            if bins >= 0.5:
                bins = 1 - bins
            bins = math.ceil(1 / bins)
            bins = len(df) // bins
        for c in df.columns:
            if pd.api.types.is_float_dtype(df[c]):
                df[c] = pd.qcut(df[c] + np.random.random(len(df))/1000, bins, labels=False)
        return df
    
    @property
    def fixed_test_indices(self):
        if self._pt_test_indices is not None:
            r = np.intersect1d(self.index.values, self._pt_test_indices)
            if len(r) < len(self._pt_test_indices):
                warnings.warn("Test rows were lost because of a previous operation.",
                    RuntimeWarning,
                )
                self._pt_test_indices = r
            return self._pt_test_indices
        return []
        
    @property
    def _test_indices(self):
        if self._pt_test_indices is not None:
            return self.fixed_test_indices
        if self._unchanged() and self._cached_test_indices is not None:
            return self._cached_test_indices
        if self._pt_folds is not None and self._test_size == 1:
            r = self._test_fold
        elif self._test_size > 0:
            if self._pt_split_stratify_test is None:
                if self._shuffle:
                    _, r = train_test_split(self._indices_before_testsplit, test_size=self._test_size, random_state=self._pt_split_random_state)
                    r = sorted(r)
                else:
                    test_size = int(self._test_size * len(self._indices_before_testsplit))
                    r = self._indices_before_testsplit[-test_size:]
            else:
                if self._pt_split_stratify_test == True or len(self._pt_split_stratify_test) > 1:
                    splitter = MultilabelStratifiedShuffleSplit(n_splits=1, 
                                                       random_state=self._pt_split_random_state, 
                                                       test_size=self._test_size)
                else:
                    splitter = StratifiedShuffleSplit(n_splits=1, 
                                                       random_state=self._pt_split_random_state, 
                                                       test_size=self._test_size) 
                target = self._stratifyable_columns(self._indices_before_testsplit, 
                                                    self._pt_split_stratify_test, self._test_size / len(self))
                _, r = next(splitter.split(target, target))
                r = sorted(r)
        else:
            r = []  
        self._change('_cached_test_indices', r)
        return r
    
    @property
    def _train_indices_unbalanced(self):
        return sorted(set(self._indices_after_testsplit) - set(self._valid_indices))

    def _pseudo_choose(self, indices, items):
        if self._pt_split_random_state is not None:
            random.seed(self._pt_split_random_state)
        r = np.array(random.sample(sorted(indices), items % len(indices)))
        r = np.hstack([indices for i in range(items // len(indices))] + [r])
        return r

    @property
    def _train_indices(self):
        if self._unchanged() and self._cached_train_indices is not None:
            return self._cached_train_indices
        r = self._train_indices_unbalanced
        if self._pt_balance is not None:
            y = self.loc[r, self._columny]
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
            r = np.array(r)[ mask.astype(int) ]
        self._change('_cached_train_indices', r)
        return r

    @property
    def _valid_indices(self):
        if self._unchanged() and self._cached_valid_indices is not None:
            return self._cached_valid_indices
        if self._pt_folds is not None:
            r = self._valid_fold
        elif self._valid_size > 0:
            if self._test_size < 1:
                valid_size = self._valid_size / (1 - self._test_size)
            else:
                valid_size = self._valid_size
            if valid_size > 0:
                if self._pt_split_stratify is None:
                    if self._shuffle:
                        _, r = train_test_split(self._indices_after_testsplit, test_size=valid_size, random_state=self._pt_split_random_state)
                        r = sorted(r)
                    else:
                        valid_size = int(valid_size * len(self._indices_before_testsplit))
                        r = self._indices_after_testsplit[-valid_size:]                        
                else:
                    if self._pt_split_stratify == True or len(self._pt_split_stratify) > 1:
                        splitter = MultilabelStratifiedShuffleSplit(n_splits=1, 
                                                           random_state=self._pt_split_random_state, 
                                                           test_size=valid_size)
                    else:
                        splitter = StratifiedShuffleSplit(n_splits=1, 
                                                           random_state=self._pt_split_random_state, 
                                                           test_size=valid_size) 
                    target = self._stratifyable_columns(self._indices_after_testsplit, self._pt_split_stratify, self._valid_size / (len(self)-self._test_size))
                    _, r = next(splitter.split(target, target))
                    r = sorted(r)
        else:
            r = []
        self._change('_cached_valid_indices', r)
        return r

    @property
    def _folds(self):
        if self._unchanged() and self._cached_folds is not None:
            return self._cached_folds
        r = []
        if 0 < self._test_size < 1:
            indices = self._indices_after_testsplit
        else:
            indices = self._indices_before_testsplit
        if self._pt_folds_stratify is None:
            target = self.loc[indices]
            splitter = KFold(n_splits = self._pt_folds, shuffle=self._pt_folds_shuffle, random_state=self._pt_folds_random_state)
            for train_indices, valid_indices in splitter.split(target, target):
                r.append(sorted(indices[valid_indices]))
        else:
            if self._pt_folds_stratify != True and len(self._pt_folds_stratify) > 1:
                splitter = MultilabelStratifiedKFold(n_splits = self._pt_folds,
                                    shuffle=True,
                                    random_state=self._pt_folds_random_state)
            else:
                splitter = StratifiedKFold(n_splits = self._pt_folds,
                                    shuffle=True,
                                    random_state=self._pt_folds_random_state)
            target = self._stratifyable_columns(indices, self._pt_folds_stratify, self._pt_folds)
            for train_indices, valid_indices in splitter.split(target, target):
                r.append(sorted(indices[valid_indices]))
        self._change('_cached_folds', r)
        return r
    
    @property
    def _fold(self):
        """
        The current valid fold number, set by df.fold(i)
        """
        try:
            return self._pt_fold + 0
        except:
            return 0
    
    @property
    def _test_fold(self):
        """
        the current test fold, determined by df.fold(i) + 1
        """
        test_fold = (self._fold + 1 + (self._fold // self._pt_folds % (self._pt_folds - 1))) % self._pt_folds
        return self._folds[test_fold]
    
    @property
    def _valid_fold(self):
        """
        The current valid fold, determined by df.fold(i)
        """
        return self._folds[self._fold]
    
    def fold(self, i):
        """
        Utilize n-fold cross validation by first calling `folds(n)` and then calling `fold(i)` to obtain 
        a DFrame in which cross validation is set up using the fold (i) as the validation set, 
        another fold as the test set and the remainder as the training set.
        
        A first call will trigger rows to be assigned to the train, valid and test part, which are stored
        in place to reproduce the exact same split for consecutive calls. The call will however return
        a copy of the DataFrame in which the requested fold is selected.

        Arguments:
            i: int
                the fold to use as the validation set
        
        Returns: copy of the PipeTorch DataFrame
            In this copy, the train, valid and test sets are shifted to fold n
        """
        self._folds
        r = self.copy(deep=False)
        
        i = i % r._pt_folds
        r._change('_pt_fold', i)
        return r
    
    def iterfolds(self):
        """
        Iterate over the folds for n-fold cross validation. 
        
        A first call will trigger rows to be assigned to the train, valid and test part, which are stored
        in place to reproduce the exact same split for consecutive calls.
        
        Yields:
            train, valid (DSet)
        """
        for i in range(self._pt_folds):
            yield self.fold(i)
    
    def df_to_dataset(self, df, datasetclass=None):
        """
        Converts the given df to a DataSet using the pipeline of this DFrame.
        
        Arguments:
            df: DataFrame or DFrame
                to convert into a DataSet
        
        returns: Converts the given df to a DataSet.
        """
        return self.df_to_dset(df).to_dataset(datasetclass)
        
    def to_datasets(self, datasetclass=None):
        """
        Prepares the train, valid and (optionally) test subsets as a DSet, which can be used to complete 
        the data preparation.
        
        A first call will trigger rows to be assigned to the train, valid and test part, which are stored
        in place to reproduce the exact same split for consecutive calls.

        Arguments:
            dataset: class (None)
                The DataSet class to use
        
        Returns: list(DataSet)
        """
        res = [ self.train.to_dataset(datasetclass) ]
        if len(self._valid_indices) > 0:
            res.append(self.valid.to_dataset(datasetclass))
        if len(self._test_indices) > 0:
            res.append(self.test.to_dataset(datasetclass))
        return res
        
    def df_to_dset(self, df):
        """
        Converts a DataFrame to a DSet that has the same pipeline as this DFrame.
        
        Arguments:
            df: DataFrame
        
        Returns: DSet
        """
        return self._dset(df)
        
    def to_databunch(self, datasetclass=None, batch_size=32, valid_batch_size=None, 
                     num_workers=0, shuffle=True, pin_memory=False, balance=False, 
                     collate=None):
        """
        Prepare the data as a Databunch that contains dataloaders for the train, valid and test part.
        
        batch_size, num_workers, shuffle, pin_memory: see Databunch/Dataloader constructor.
        
        A first call will trigger rows to be assigned to the train, valid and test part, which are stored
        in place to reproduce the exact same split for consecutive calls.

        Returns: Databunch
        """
        return Databunch(self, *self.to_datasets(datasetclass=datasetclass), 
                         batch_size=batch_size, valid_batch_size=valid_batch_size, 
                         num_workers=num_workers, shuffle=shuffle, 
                         pin_memory=pin_memory, balance=balance, collate=collate)    

    def _evaluator(self, *metrics):
        """
        Creates a PipeTorch Evaluator, that can be used to visualize the data, the results
        of a model or cache learning/validation diagnostics. Since datasets are often
        reused in repeated experiments, every call will create a new Evaluator to prevent
        mixing results from different experiments. If you do want to compare results from
        multiple experiments, create a single evaluator and reuse that for the experiments
        that you wish to compare.
        
        Arguments:
            *metrics: callable
                One or more functions, that will take y_true, y_pred as parameter to
                compute an evaluation metric. Typically, functions from SKLearn.metrics 
                can be used.
        
        returns: Evaluator
        """
        return Evaluator(self, *metrics)
   
    def evaluator(self, *metrics):
        try:
            if self._pt_evaluator.metrics == metrics:
                return self._pt_evaluator
        except: pass
        self._pt_evaluator = self._evaluator(*metrics)
        return self._pt_evaluator

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
            return self._dset(self.loc[indices], transforms=transforms)
        else:
            try:
                low, high = min(indices), max(indices) + self._sequence_window + self._shift_y - 1
                return self._dset(self.loc[low:high], transforms=transforms)
            except:
                return self._dset(self.loc[:0], transforms=transforms)
    
    @property
    def train(self):
        """
        Prepares the train subset as a DSet and optionally trains transformation parameters, e.g. for image
        normalization or text tokenization. Transformations are only supported for PyTorch DataSets.
        
        A first call will trigger rows to be assigned to the train, valid and test part, which are stored
        in place to reproduce the exact same split for consecutive calls.
        
        Returns: DSet
        """
        if self._unchanged() and self._cached_train is not None:
            return self._cached_train
        
        r = self._dset_indices(self._train_indices, self._dataset_transforms())
        if self._dataset_train_transformation_parameters(r):
            r = self._dset_indices(self._train_indices, self._dataset_transforms())
        self._change('_cached_train', r)
        self._cached_raw_train = None
        return r
                    
    @property
    def raw_train(self):
        if self._unchanged() and self._cached_raw_train is not None:
            return self._cached_raw_train
        self._cached_raw_train = self._dset_indices(self._train_indices, self._dataset_transforms())
        return self._cached_raw_train
    
    @property
    def valid(self):
        if self._unchanged() and self._cached_valid is not None:
            return self._cached_valid
        
        r = self._dset_indices(self._valid_indices, self._dataset_transforms(train=False))
        self._change('_cached_valid', r)
        return r

    @property
    def test(self):
        if self._unchanged() and self._cached_test is not None:
            return self._cached_test
        
        if self._pt_sequence_window is None:
            r = DSet.df_to_testset(self.loc[self._test_indices], self, self._dataset_transforms(train=False))
        else:
            low, high = min(self._test_indices), max(self._test_indices) + self._sequence_window + self._shift_y - 1
            r = DSet.df_to_testset(self.loc[low:high], self, self._dataset_transforms(train=False))
        self._change('_cached_test', r)
        return r
    
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
    
    def numpy(self):
        """
        Returns: train_X, train_y, valid_X, valid_y
        """
        return self.train_X, self.train_y, self.valid_X, self.valid_y
        
    def tensor(self):
        return self.train.X_tensor, self.train.y_tensor, self.valid.X_tensor, self.valid.y_tensor
    
    @property
    def test_y(self):
        return self.test.y
    
    def loss_surface(self, model, loss, **kwargs):
        self._evaluator(loss).loss_surface(model, loss, **kwargs)
    
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
        Intended for internal use. Adds a column with values for the target variable to the DataFrame. 
        When applicable, the transformation for the target variable is automatically inverted. 
        This is useful to evaluate or visualize results.
        
        This effect is not inplace, but configured to a copy that is returned. 
        
        Arguments:
            y: Numpy/PyTorch array
                values in the same for as those generated for the target variable 
                (possibly predictions) that will be added to the DataFrame
            erase_y: bool (True)
                whether the original target variable is removed 
        
        Returns: DFrame 
        """
        df_y = self.inverse_scale_y(y)
        r = self.copy()
        if columns is None:
            columns = [ c + '_pred' for c in self._columny ]
        r[columns] = np.NaN
        r.loc[r.index[indices], columns] = df_y.values
        r = self._dframe(r)
        r._cached_changed = True
        return r
    
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
        self._evaluator().plot_boundary(predict)

    def balance(self, weights=True):
        """
        Oversamples rows in the training set, so that the values of the target variable 
        are better balanced. Does not affect the valid/test set.
        
        This effect is not inplace, but configured to a copy that is returned. 

        Arguments:
            weights: True or dict
                when set to True, the target values of the training set are 
                uniformely distributed,
                otherwise a dictionary can be passed that map target values to the 
                desired fraction of the training set (e.g. {0:0.4, 1:0.6}).
        
        Returns: DFrame
        """
        r = self.copy(deep=False)
        r._change('_pt_balance', weights)
        return r    
  
    def scale(self, columns=True, scalertype=StandardScaler, omit_interval=(-2,2)):
        """
        Scales the features and target variable in the DataFrame. A scaler is fitted on the
        train data and applied to train, valid and test.
        
        This effect is not inplace, but configured to a copy that is returned. 

        Arguments:
            columns: True, str or list of str (True)
                the columns to scale (True for all)
                
            scalertype: a SKLearn type scaler (StandardScaler)
                the scaler class that is used
            
            omit_interval: (-2,2) when colums is set to True
                features whose values lie within this interval are not scaled, the default value
                ensures that binary values will not be scaled.
                
        Return: DFrame
        """
        if self._pt_polynomials and columns != 'x_only':
            assert type(columns) != list or len(columns) == 0, 'You cannot combine polynomials with column specific scaling'
        r = self.copy(deep=False)
        r._change('_pt_scale_columns', columns)
        r._change('_pt_scalertype', scalertype)
        r._change('_pt_scale_omit_interval', omit_interval)
        return r
    
    def scalex(self, scalertype=StandardScaler, omit_interval=(-2,2)):
        """
        Scale all input features. A scaler is fitted on the
        train data and applied to train, valid and test.
        
        This effect is not inplace, but configured to a copy that is returned. 

        Arguments:
            scalertype: a SKLearn scaler class (StandardScaler)
                the scaler class that is used
                
            omit_interval: tupple (-2,2) 
                features whose values lie within this interval are not scaled, the default value
                ensures that binary values will not be scaled.
        
        Returns: DFrame 
        """
        return self.scale(columns='x_only', scalertype=scalertype, omit_interval=omit_interval)
    
    def add_bias(self):
        """
        Adds a bias column of value 1 to generated inputs.
        
        This effect is not inplace, but configured to a copy that is returned. 

        Returns: DFrame
        """
        r = self.copy(deep=False)
        r._change('_pt_bias', True)
        return r
    
    def reshuffle(self, random_state=None):
        """
        Resamples the train, valid and test set with the existing settings.
        
        This effect is not inplace, but applied to a copy that is returned. 

        Arguments:
            random_state: int (None)
                set a random_state for reproducible results   
                
        Returns: DFrame
        """
        r = self.copy(deep=False)
        r._change('_pt_split_random_state', random_state)
        return r
    
    def _index_changed(self):
        for p in self._cached_indices:
            setattr(self, p, None)
        for p in self._cached_fold:
            setattr(self, p, None)
        for p in self._cached:
            setattr(self, p, None)
        self._cached_index = self.index.copy()

    def _columns_changed(self):
        for p in self._cached:
            setattr(self, p, None)
    
    def reset_indices(self):
        """
        Clears the currently sampled split() and folds(), which is stored whenever data preparation
        is called. Therefore, resampling the split or fold on the next call to data preparation.
        However, this call will not reset a random_state, use reshuffle() for that.
        
        This effect is not inplace, but applied to a copy that is returned. 
        
        Returns: DFrame
        """
        r = self.copy(deep=False)
        r._index_changed()
        return r
        
    def split(self, 
              valid_size=None, 
              test_size=None, 
              shuffle=None, 
              random_state=None, 
              stratify=None, 
              stratify_test=None):
        """
        Split the data in a train/valid/(test) set. 
        
        This effect is not inplace, but applied to a copy that is returned. 
        
        Arguments:
            valid_size: float (None)
                the fraction of the dataset that is used for the validation set.
                
            test_size: float (None)
                the fraction of the dataset that is used for the test set. When combined with folds
                if 1 > test_size > 0, the test set is split before the remainder is divided in folds 
                to apply n-fold cross validation.
                
            shuffle: bool (None)
                shuffle the rows before splitting. None means True unless sequence() is called to process
                the data as a (time) series.
                
            random_state: int (None)
                set a random_state for reproducible results
                
            stratify: str, [ str ], True or None (None)
                Apply stratified sampling over these columns (True = all columns)
                For multiple columns, multi-label stratification is applied with support over
                continuous variables.
                
            stratify_test: str, [ str ], True or (None)
                Apply stratified sampling over these columns (True = all columns)
                For multiple columns, multi-label stratification is applied with support over
                continuous variables.
                If None, the value for stratify is used. To supress stratification, pass [].
            
        Returns: DFrame 
        """
        r = self.copy(deep=False)
        r._change('_pt_valid_size', valid_size)
        r._change('_pt_test_size', test_size)
        r._change('_pt_split_shuffle', shuffle)
        r._change('_pt_split_random_state', random_state)
        r._change('_pt_split_stratify', [stratify] if type(stratify) == str else stratify)
        r._change('_pt_split_stratify_test', [stratify_test] if type(stratify_test) == str else \
                                    (r._pt_split_stratify if stratify_test is None else stratify_test))
        return r
    
    def folds(self, folds=5, shuffle=True, random_state=None, stratify=None, test=None):
        """
        Divide the data in folds to setup n-Fold Cross Validation in a reproducible manner. 
        
        By combining folds() with split(0 < test_size < 1) , a single testset is split before 
        dividing the remainder in folds that are used for training and validation. 
        When used without split, by default a single fold is used for testing.
        
        The folds assigned to the validation and test-set rotate differently, 
        giving 5x4 combinations for 5-fold cross validation. You can access all 20 combinations
        by calling fold(0) through fold(19).
    
        This effect is not inplace, but applied to a copy that is returned. 
    
        Arguments:
            folds: int (None)
                The number of times the data will be split in preparation for n-fold cross validation. The
                different splits can be used through the fold(n) method.
                SKLearn's SplitShuffle is used, therefore no guarantee is given that the splits are
                different nor that the validation splits are disjoint. For large datasets, that should not
                be a problem.
                
            shuffle: bool (None)
                shuffle the rows before splitting. None means True unless sequence() is called to process
                the data as a (time) series.
                
            random_state: int (None)
                set a random_state for reproducible results.
                
            stratify: str, [ str ], True or None (None)
                Apply stratified sampling over the given columns. True means all columns. If column 
                Per value for the given column, the rows are sampled. When a list
                of columns is given, multi-label stratification is applied.
                
            test: bool (None)
                whether to use one fold as a test set. The default None is interpreted as True when
                split is not used. Often for automated n-fold cross validation studies, the validation set
                is used for early termination, and therefore you should use an out-of-sample
                test set that was not used for optimizing.
            
        Returns: copy of DFrame 
            schedules the data to be split in folds.
        """
        assert type(folds) == int and folds > 1, 'You have to set split(folds) to an integer > 1'
        r = self.copy(deep=False)
        r._change('_pt_folds', folds)
        r._change('_pt_folds_shuffle', shuffle)
        r._change('_pt_folds_random_state', random_state)
        r._change('_pt_folds_stratify', [stratify] if type(stratify) == str else stratify)
        if test or (r._pt_test_size is None and test is None):
            r._change('_pt_test_size', 1)
        return r
    
    def leave_one_out(self):
        """
        Configures folds() to perform a leave-one-out cross validation.
        """
        return self.folds(len(self._indices_after_testsplit), shuffle=False)
        
    def polynomials(self, degree, include_bias=False):
        """
        Adds (higher-order) polynomials to the data pipeline.
        
        This effect is not inplace, but configured to a copy that is returned. 
        
        Arguments:
            degree: int - degree of the higher order polynomials (e.g. 2 for squared)
            include_bias: bool (False) - whether to generate a bias column
        
        Returns: copy of DFrame 
        """
        assert type(self._pt_scale_columns) != list or len(self._pt_scale_columns) == 0, 'You cannot combine polynomials with column specific scaling'
        r = self.copy(deep=False)
        r._change('_pt_polynomials', PolynomialFeatures(degree, include_bias=include_bias))
        return r
    
    def no_columny(self):
        """
        PipeTorch cannot currently handle a dataset without a target variable, 
        however, it can work by simply assigning one of the used input features
        as a target variable.

        This effect is not inplace, but configured to a copy that is returned. 

        Returns: copy of DFrame
        """
        r = self.copy(deep=False)
        r._change('_pt_columny', [self._pt_columnx[0]])
        return r
    
    def columny(self, columns=None, vector=None):
        """
        Configures the generated target variable and shape.
        
        This effect is not inplace, but configured to a copy that is returned. 

        Arguments:
            columns: str or list of str (None)
                single column name or list of columns that is to be used as target column. 
                None: use the last column
            vector: bool (None)
                By default, y will be prepared as a 1d vector and y_tensor as an (n, 1) matrix
                because most SKLearn algorithmms prefer vector and for PyTorch (n, 1) matrices 
                are used more often. Setting vector=False causes both y and y_tensor to return 1d shapes
                and vector=True causes both y and y_tensor to return (n, 1) shapes.
        
        Returns: DFrame 
        """
        
        r = self.copy(deep=False)
        if columns is not None:
            r._change('_pt_columny', [columns] if type(columns) == str else columns)
        assert r._pt_columny is None or len(r._pt_columny) == 1 or not vector, 'You cannot create target vector with multiple columns'
        r._change('_pt_vectory', vector)
        return r

    def columnx(self, *columns, omit=False):
        """
        Specify which columns to use as input. The target variable is always excluded
        so if you want to add that (for sequence learning), you should copy the
        target column. 
        Unlocks the DFrame, train and valid sets are therefore resampled.
        
        This effect is not inplace, but configured to a copy that is returned. 

        Arguments:
            columns: str or list of str
                columns to be used as input features
            omit: bool (False)
                when True, all columns are used except the specified target column(s)
        
        Returns: DFrame
        """
        r = self.copy(deep=False)
        if omit:
            r._change('_pt_columnx', [ c for c in self.columns if c not in columns ])
        else:
            r._change('_pt_columnx', list(columns) if len(columns) > 0 else None)
        return r
    
    def category(self, *columns, sort=False):
        """
        Converts the values in the targetted columns into indices, for example to use in lookup tables.
        columns that are categorized are excluded from scaling. You cannot use this function together
        with polynomials or bias.
        
        Note: PipeTorch only uses categories that are in the training set and uses category 0 as
        an unknown category number for categories in the validation and test set that are not known during training.
        This way, no future information is used. 
        
        This effect is not inplace, but configured to a copy that is returned. 

        Arguments:
            columns: str or list of str
                list of columns that is to be converted into a category
            sort: True/False (default False) 
                whether the unique values of these colums should be converted 
                to indices in sorted order.
        
        Returns: DFrame
        """
        assert self._pt_polynomials is None, 'You cannot combine categories with polynomials'
        assert self._pt_bias is None, 'You cannot combine categories with polynomials'
        r = self.copy(deep=False)
        r._change('_pt_category', columns)
        r._change('_pt_category_sort', sort)
        return r
    
    def dummies(self, *columns):
        """
        Converts the values in the targetted columns into dummy variables. This is an alernative to 
        pd.get_dummies, that only uses the train set to assess which values there are (as it should be),
        setting all variables to 0 for valid/test items that contain an unknown label. 
        That way, no future information is used. Columns that are categorized are excluded from scaling. 
        You cannot use this function together with polynomials or bias.
        
        This effect is not inplace, but configured to a copy that is returned. 

        Args:
            columns: str or list of str
                the columns that are to be converted into a category
        
        Returns: DFrame 
        """
        assert self._pt_polynomials is None, 'You cannot combine categories with polynomials'
        assert self._pt_bias is None, 'You cannot combine categories with polynomials'
        r = self.copy(deep=False)
        r._change('_pt_dummies', columns)
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
        
        This effect is not inplace, but configured to a copy that is returned. 

        Arguments:
            window: int
                the number of rows that is used a an input
            shift_y: int
                how many rows into the future the target variable is placed, 
                e.g. with a window=2 and shift_y=1, X0 would contain [x[0], x[1]] 
                while y0 would contain y[2], 
                the next sample X1 would be [x[1], x[2]] and y[3].
        
        Returns: DFrame
        """
        r = self.copy(deep=False)
        r._change('_pt_sequence_window', window)
        r._change('_pt_sequence_shift_y', shift_y)
        return r
        
    def filterna(self, filter=True):
        """
        Returns: a DataFrame in which rows with missing values are filtered
        from the train and valid set. The result is not visible in the DataFrame
        but will be in the resulting train and valid sets.
        
        Args:
            filter: bool (True)
                Default is to drop rows in de DSet's that still have missing values.
                Set to false to turn off
        """
        r = self.copy(deep=False)
        r._change('_pt_filterna', filter)
        return r
    
    def columnfunction(self, **columnfunctions):
        """
        Returns: a DataFrame in which column transformations are added to the
        data pipeline, before scaling the data.
        
        Transforms effectively adds to the existing configured column functions,
        while overriding functions that are newly defined. This function does not
        remove any column functions, unless called with zero parameters
        
        Args:
            columnfunctions: { columname: callable}
                when empty, all existing column functions will be erased,
                otherwise, the given column functions are added to the dictionary
                of configured column functions, which are executed in the pipeline
                just before str column names and a function
                that will be performed on the indicated column.
        """

        assert all([ c in self.columns for c in columnfunctions ]), 'Not all keys are columns'
        r = self.copy(deep=False)
        if r._pt_transform is None:
            r._change('_pt_transform', columnfunctions)
        else:
            r._change('_pt_transform', r._pt_transform | columnfunctions)
        return r
    
    def log(self, *columns):
        """
        Returns: a DataFrame in which log column transformations are added to the
        data pipeline, before scaling the data.
        
        Args:
            columnfunctions: { columname: callable}
                a dictionary of str column names and a function
                that will be performed on the indicated column.
        """
        return self.columnfunction( **{c:FunctionTransformer(np.log) for c in columns })
        
    def log1p(self, *columns):
        """
        Returns: a DataFrame in which log1p column transformations are added to the
        data pipeline, before scaling the data.
        
        Args:
            columnfunctions: { columname: callable}
                a dictionary of str column names and a function
                that will be performed on the indicated column.
        """
        return self.columnfunction( **{c:FunctionTransformer(np.log1p) for c in columns })
    
    def dataset_train_transforms(self, *transforms):
        """
        Configure a (list of) transformation function(s) that is called from the DataSet class to prepare the 
        train data.
        
        This effect is not inplace, but configured to a copy that is returned. 

        Arguments:
            *transforms: [ callable ]
                (list of) transformation function(s) that is called from the DataSet class to prepare the data.

        Returns: DFrame
        """
        r = self.copy(deep=False)
        r._change('_pt_dataset_train_transforms', transforms)
        return r
    
    def dataset_transforms(self, *transforms):
        """
        Configure a (list of) transformation function(s) that is called when retrieving
        items from PipeTorch' TransformableDataSet class to prepare the data. These function allow
        to configure a pipeline for data augmentation that is used to train PyTorch models.
        For example, to train of images or audio fragments that are read from disk.
        
        Note that these transformations are not used to prepare Numpy Arrays, and the DataSet class has
        to call the tranformations (which the TransformableDataSet class does that is used by default).

        This effect is not inplace, but configured to a copy that is returned. 

        Arguments:
            *transforms: [ callable ]
                (list of) transformation function(s) that is called from the DataSet class to prepare the data.
                
        Returns: DFrame
        """
        r = self.copy(deep=False)
        r._change('_pt_dataset_transforms', transforms)
        return r

    def _dataset_train_transformation_parameters(self, train_dset):
        """
        Placeholder to extend DFrame to configure transformations that are applied on the train DataSet,
        but not on the validation or test set. This is used by
        ImageDFrame to learn normalization parameters for images and to learn a vocabulary 
        for TextDFrame to tokenize text.
        
        Note that the mechanism assumes that the train DSet is generated first.
        
        Arguments: 
            train_dset: DSet
                The train DSet to learn the transformation parameters on
            
        Returns: bool (None)
            This function should return True when parameters were learned, to force DFrame to add a new
            list of transformations to the generated DSet.
        """
        pass
    
    def _dataset_pre_transforms(self):
        """
        Placeholder for standard transformations that precede configured transformations.
        """
        return []
    
    def _dataset_post_transforms(self):
        """
        Placeholder for standard transformations that succede configured transformations.
        """
        return []
    
    def _dataset_transforms(self, pre=True, train=True, standard=True, post=True):
        """
        Arguments:
            pre: bool (True) - whether to include pre transformations
            train: bool (True) - whether to include train transformations
            standard: bool (True) - whether to include standard transformations
            post: bool (True) - whether to include post transformations
            
        Returns: [ callable ]
            A list of transformations that is applied to the generated DSet 
        """
        t = []
        try:
            if pre:
                t.extend(self._dataset_pre_transforms())
        except: pass
        try:
            if train:
                t.extend(self._pt_dataset_train_transforms)
        except: pass
        try:
            t.extend(self._pt_dataset_transforms)
        except: pass
        try:
            if post:
                t.extend( self._dataset_post_transforms() )
        except: pass
        return t
    
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

    def cross_validate_sklearn(self, model, *target, evaluator=None, annot={}, **kwargs):
        """
        On a DFrame that is configured for n-fold cross validation (df.folds(n)), this function iterates
        over the trials, fitting an SKLearn model and storing the targets for the train, valid and test subsets.
        
        Arguments:
            model: object
                a machine learning algorithm that support the fit(X, y) and predict(X) methods,
                like the SKLearn models.
            target: callable
                one or more functions that return an evaluation metrics when called with
                target(y_true, y_predict)
            evaluator: Evaluator (None)
                when provided, the results are added to this evaluator, otherwise
                the evaluator of this DFrame is reset and a new one is used.
            annot: {}
                the annotations that are stored with the metrics. cross_validate will add a
                'fold' metric to indicate the fold.
                
        """
        assert self._pt_folds is not None, 'You have to set df.folds before you can use cross validate'
        assert self._pt_folds > 1, 'You have to set df.folds greater than 1 before you can use cross validate'
        if reset_evaluator:
            try:
                del self._pt_evaluator
            except: pass
        evaluator = self.evaluator(*target)
        study = evaluator.study(**kwargs)
        data = self.iterfolds()
        folds = self._pt_folds
        test = len(self._test_indices) > 0
                
        def run(evaluator, trial):
            df = next(data)
            annot['fold'] = trial.number
            model.fit(df.train_X, df.train_y)
            evaluator._store_metrics(df.train_y, model.predict(df.train_X), 
                                          annot={'phase':'train', **annot})
            metrics = evaluator._store_metrics(df.valid_y, model.predict(df.valid_X), 
                                                    annot={'phase':'valid', **annot})
            if test:
                metrics = evaluator._store_metrics(df.test_y, model.predict(df.test_X), 
                                              annot={'phase':'test', **annot})
            
            return [ metrics[t] for t in study.target ]
        
        study.optimize(run, n_trials=folds)
        return study

    def study(self, *target, evaluator=None, **kwargs):
        """
        On a DFrame that is configured for n-fold cross validation (df.folds(n)), this function iterates
        over the trials, fitting an SKLearn model and storing the targets for the train, valid and test subsets.
        
        Arguments:
            target: callable
                one or more functions that return an evaluation metrics when called with
                target(y_true, y_predict)
            evaluator: Evaluator (None)
                when provided, the results are added to this evaluator, otherwise
                the evaluator of this DFrame is reset and a new one is used.
        """
        if evaluator is None:
            try:
                del self._pt_evaluator
            except: pass
            evaluator = self.evaluator(*target)
        study = evaluator.study(**kwargs)
        return study

class DFrame(pd.DataFrame, _DFrame):
    _metadata = _DFrame._metadata
    _internal_names = _DFrame._internal_names
    _internal_names_set = _DFrame._internal_names_set

    def __init__(self, data, *args, **kwargs):
        super().__init__(data, *args, **kwargs)
        _DFrame.__init__(self, data)
        self._catch_change()

    def _changed(self, f, *args, **kwargs):
        #print(f'changed {f} {args} {kwargs}')
        self._columns_changed()
        return f(*args, **kwargs)

    def _return_changed(self, f, *args, **kwargs):
        r = f(*args, **kwargs)
        r._columns_changed()
        return r

    def _change(self, key, value):
        #print(f'changed_attr {key} {value}')
        if key in self._config_indices_set:
            self._index_changed()
        elif key in self._config_set:
            self._columns_changed()
        return super().__setattr__(key, value)

    def __changed_inplace(self, **kwargs):
        try:
            if kwargs['inplace']:
                if r._unchanged():
                    r._columns_changed()
        except: pass
        try:
            if kwargs['copy'] == False:
                if r._unchanged():
                    r._columns_changed()
        except: pass

    
    def _changed_inplace(self, f, *args, **kwargs):
        #print(f'changed_inplace {f} {args} {kwargs}')
        self.__changed_inplace(**kwargs)
        r = f(*args, **kwargs)
        r._columns_changed()
        return r
     
    def _unchanged(self):
        #print(f'process_changed {self._cached_changed} {self.index.equals(self._cached_index)}')
        
        if self._cached_index is not None and self.index.equals(self._cached_index):
            return True
        self._index_changed()
        return False
    
    def _catch_change_f(self, f):
        return functools.wraps(f)(functools.partial(self._changed, f))
    
    def _catch_return_change_f(self, f):
        return functools.wraps(f)(functools.partial(self._return_changed, f))
    
    def _catch_change_inplace_f(self, f):
        return functools.wraps(f)(functools.partial(self._changed_inplace, f))
    
    def _catch_change(self):
        self._set_value = self._catch_change_f(self._set_value)
        self._setitem_slice = self._catch_change_f(self._setitem_slice)
        self._setitem_frame = self._catch_change_f(self._setitem_frame)
        self._setitem_array = self._catch_change_f(self._setitem_array)
        self._set_item = self._catch_change_f(self._set_item)
        self.__getitem__ = self._catch_return_change_f(self.__getitem__)
        self.update = self._catch_change_f(self.update)
        self.insert = self._catch_change_f(self.insert)

        self._mgr.idelete = self._catch_change_f(self._mgr.idelete)
        self._mgr.iset = self._catch_change_f(self._mgr.iset)
        self._mgr.insert = self._catch_change_f(self._mgr.insert)
        #self._mgr.column_setitem = self._catch_change_f(self._mgr.column_setitem)
        #self._mgr.reindex_axis = self._catch_change_f(self._mgr.reindex_axis)
        #self._mgr.reindex_indexer = self._catch_change_f(self._mgr.reindex_indexer)

        self.clip = self._catch_change_inplace_f(self.clip)
        self.drop = self._catch_change_inplace_f(self.drop)
        self.drop_duplicates = self._catch_change_inplace_f(self.drop_duplicates)
        #self.dropna = self._catch_change_inplace_f(self.dropna)
        self.eval = self._catch_change_inplace_f(self.eval)

        self.fillna = self._catch_change_inplace_f(self.fillna)
        self.interpolate = self._catch_change_inplace_f(self.interpolate)
        self.mask = self._catch_change_inplace_f(self.mask)
        self.pad = self._catch_change_inplace_f(self.pad)
        self.query = self._catch_change_inplace_f(self.query)
        self.replace = self._catch_change_inplace_f(self.replace)
        self.reset_index = self._catch_change_inplace_f(self.reset_index)
        self.set_axis = self._catch_change_inplace_f(self.set_axis)
        self.set_index = self._catch_change_inplace_f(self.set_index)
        self.sort_index = self._catch_change_inplace_f(self.sort_index)
        self.sort_values = self._catch_change_inplace_f(self.sort_values)
        self.where = self._catch_change_inplace_f(self.where)
        self.astype = self._catch_change_inplace_f(self.astype)
        self.reindex = self._catch_change_inplace_f(self.reindex)
        self.rename = self._catch_change_inplace_f(self.rename)
        #self.__setattr__ = self._changed_attr

    def dropna(self, **kwargs):
        if self._pt_test_indices is not None:
            warnings.warn("Using dropna on a dataset with a fixed test set " +
                    "which is probably a bad idea since this removes test data.",
                    RuntimeWarning,
                    stacklevel=find_stack_level(),
                )
        self.__changed_inplace(**kwargs)
        return pd.DataFrame.dropna(self, **kwargs)

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
    
    @classmethod
    def read_from_kaggle(cls, dataset, train=None, test=None, shared=True, force=False, **kwargs):
        """
        Reads a DFrame from a Kaggle dataset. The downloaded dataset is automatically stored so that the next time
        it is read from file rather than downloaded. See `read_csv`. The dataset is stored by default in a folder
        with the dataset name in `~/.pipetorchuser`.

        If the dataset is not cached, this functions requires a valid .kaggle/kaggle.json file, that you can 
        create manually or with the function `create_kaggle_authentication()`.

        Note: there is a difference between a Kaggle dataset and a Kaggle competition. For the latter, 
        you have to use `read_from_kaggle_competition`.

        Example:
            read_from_kaggle('uciml/autompg-dataset')
                to read/download `https://www.kaggle.com/datasets/uciml/autompg-dataset`
            read_from_kaggle('robmarkcole/occupancy-detection-data-set-uci', 'datatraining.txt', 'datatest.txt')
            to combine a train and test set in a single DFrame

        Arguments:
            dataset: str
                the username/dataset part of the kaggle url, e.g. uciml/autompg-dataset for 

            train: str (None)
                the filename that is used as the train set, e.g. 'train.csv'
            test: str (None)
                the filename that is used as the test set, e.g. 'test.csv'
            shared: bool (False)
                save the dataset in ~/.pipetorch instead of ~/.pipetorchuser, allowing to share downloaded
                files between users.
            force: bool (False)
                when True, the dataset is always downloaded
            **kwargs:
                additional parameters passed to pd.read_csv. For example, when a multichar delimiter is used
                you will have to set engine='python'.

        Returns: DFrame
        """
        k = Kaggle(dataset, shared=shared)
        if force:
            k.remove_user()
        train = k.read(train, **kwargs)
        if test is not None:
            test = k.read(test, **kwargs)
            return cls.from_train_test(train, test)
        return cls(train)
           
#     @classmethod
#     def read_from_kaggle_competition(cls, dataset, train=None, test=None, shared=False, force=False, **kwargs):
#         train = read_from_kaggle_competition(dataset, filename=train, shared=shared, force=force, **kwargs)
#         if test is not None:
#             test = read_from_kaggle_competition(dataset, filename=test, **kwargs)
#             return cls.from_train_test(train, test)
#         return cls(train)

#     @classmethod
#     def read_csv(cls, url, filename=None, path=None, save=False, **kwargs):
#         """
#         Reads a .csv file from cache or url. The place to store the file is indicated by path / filename
#         and when a delimiter is used, this is also used to save the file so that the original delimiter is kept.
#         The file is only downloaded using the url if it does not exsists on the filing system. If the file is
#         downloaded and save=True, it is also stored for future use.

#         Arguments:
#             url: str
#                 the url to download or a full path pointing to a .csv file
#             filename: str (None)
#                 the filename to store the downloaded file under. If None, the filename is extracted from the url.
#             path: str (None)
#                 the path in which the file is stored. If None, it will first check the ~/.pipetorch (for sharing
#                 dataset between users) and then ~/.pipetorchuser (for user specific caching of datasets).
#             save: bool (False)
#                 whether to save a downloaded .csv
#             **kwargs:
#                 additional parameters passed to pd.read_csv. For example, when a multichar delimiter is used
#                 you will have to set engine='python'.

#         Returns: DFrame
#         """
#         return cls(read_csv(url, filename=filename, path=path, save=save, **kwargs))
   
    @classmethod
    def read_from_package(cls, package, filename, **kwargs):
        return cls(read_from_package(package, filename, **kwargs))
    
    @classmethod
    def read_from_function(cls, filename, function, path=None, save=True, **kwargs):
        """
        First checks if a .csv file is already stored, otherwise, calls the custom function to retrieve a 
        DataFrame. 

        The place to store the file is indicated by path / filename.
        The file is only retrieved from the function if it does not exsists on the filing system. 
        If the file is retrieved and save=True, it is also stored for future use.

        Arguments:
            filename: str (None)
                the filename to store the downloaded file under.
            function: func
                a function that is called to retrieve the DataFrame if the file does not exist.
            path: str (None)
                the path in which the file is stored. If None, it will first check the ~/.pipetorch (for sharing
                dataset between users) and then ~/.pipetorchuser (for user specific caching of datasets).
            save: bool (True)
                whether to save a downloaded .csv
            **kwargs:
                additional parameters passed to pd.read_csv. For example, when a multichar delimiter is used
                you will have to set engine='python'.

        Returns: DFrame
        """
        return cls(read_from_function(filename, function, path=path, save=save, **kwargs))

    @classmethod
    def read_excel(cls, path, filename=None, **kwargs):
        return cls(read_excel(path, filename=filename, **kwargs))

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
    
    def _unchanged(self):
        return True
    
    def _index_changed(self):
        pass
        
    def _columns_changed(self):
        pass
    
    def get_group(self, namel, obj=None):
        return self._dframe( super().get_group(name, obj=obj) )
        
    def __iter__(self):
        for group, subset in super().__iter__():
            yield group, self._copy_meta(subset)
        
    def to_dataset(self):
        """
        Convert a grouped DFrame into a PyTorch ConcatDataset over datasets
        for every group contained.
        
        returns: train, valid, test DataSet
        """
        from torch.utils.data import ConcatDataset
        dss = []
        for key, group in self:
            dss.append( self.df_to_dataset(group) )

        return [ConcatDataset(ds) for ds in zip(*dss)]
