import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, PolynomialFeatures, OneHotEncoder
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
from .helper import read_from_kaggle, read_from_kaggle_competition, read_csv, read_from_package, read_from_function
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
                 '_pt_vectory', '_pt_bias', '_pt_polynomials', '_pt_dtype', '_pt_category', '_pt_category_sort', 
                 '_pt_dummies', '_pt_sequence_window', '_pt_sequence_shift_y', 
                 '_pt_split_shuffle', '_pt_split_stratify', '_pt_split_random_state', 
                 '_pt_folds_shuffle', '_pt_folds_stratify', '_pt_folds_random_state', 
                 '_pt_valid_size', '_pt_test_size', '_pt_balance', 
                 '_pt_train_valid_indices', '_pt_valid_indices', '_pt_test_indices',
                 '_pt_folds', '_pt_fold', '_pt_created_folds', 
                 '_pt_dataset', '_pt_transforms', '_pt_train_transforms']

    _locked_names = [  '_pt__locked_train_indices', 
                       '_pt__locked_valid_indices', '_pt__locked_test_indices',
                       '_pt__locked_train', '_pt__locked_valid', 
                       '_pt__locked_scalerx', '_pt__locked_scalery',
                       '_pt__locked_categoryx', '_pt__locked_categoryy', 
                       '_pt__locked_dummiesx', '_pt__locked_dummiesy', '_pt__locked_len',
                       '_pt__indices_before_testsplit', '_pt__indices_after_testsplit',
]

    _internal_names = pd.DataFrame._internal_names + _locked_names
    
    _internal_names_set = set( _internal_names )
    
    def __init__(self, data, **kwargs):
        for m in self._metadata:
            self.__setattr__(m, None)
            self._pt_vectory = False
            self._pt_fold = 0
            try:
                self.__setattr__(m, getattr(data, m))
            except: pass

    def _copy_meta(self, r):
        for c in self._metadata:
            setattr(r, c, getattr(self, c))
        return r
            
    def _copy_indices(self, r):
        if self.is_locked:
            r._pt__locked_train_indices = self._pt__locked_train_indices
            r._pt__locked_valid_indices = self._pt__locked_valid_indices
            r._pt__locked_test_indices = self._pt__locked_test_indices
            r._pt__locked_len = self._pt__locked_len
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
            return self._pt__locked_train_indices is not None and len(self) == self._pt__locked_len
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
            self._pt__locked_valid_indices = self._valid_indices
            self._pt__locked_train_indices = self._train_indices
            self._pt__locked_test_indices = self._test_indices
            self._pt__locked_len = len(self)
    
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
        r = self._copy_with_indices()
        r._pt_dtype = dtype
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
    
    def _indices_notnull(self):
        if self._pt_sequence_window is not None:
            def a(w):
                return np.all(w)

            indicesy = np.where(self[self._columny].notnull().all(1))[0]
            indicesx = self[self._columnx].notnull().all(1)[::-1].rolling(self._pt_sequence_window).apply(a, raw=True, engine='numba')[::-1].index
            indicesy = set(indicesy - (self._pt_sequence_window + self._pt_sequence_shift_y - 1))
            return np.array([ i for i in indicesx if i in indicesy ])
        else:
            return np.array(np.where(self[self._all_columns].notnull().all(1))[0])
    
    @property
    def _indices_unshuffled(self):
        try:
            return np.array([ i for i in self._pt_train_valid_indices if i in set(self._indices_notnull()) ])
        except:
            return self._indices_notnull()

    @property
    def _indices_before_testsplit(self):
        try:
            if self._pt__indices_before_testsplit is not None:
                return self._pt__indices_before_testsplit
        except: pass
        try:
            test_indices = set(self._test_indices)
            self._pt__indices_before_testsplit = np.array([ i for i in self._indices_unshuffled if i not in test_indices ])
        except:
            self._pt__indices_before_testsplit = self._indices_unshuffled
        return self._pt__indices_before_testsplit
           
    @property
    def _indices_after_testsplit(self):
        try:
            if self._pt__indices_after_testsplit is not None:
                return self._pt__indices_after_testsplit
        except: pass
        test_indices = set(self._test_indices)
        self._pt__indices_after_testsplit = np.array([ i for i in self._indices_before_testsplit if i not in test_indices ])
        return self._pt__indices_after_testsplit

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
        try:
            del self._pt_valid_indices
        except: pass
        try:
            del self._pt_train_indices
        except: pass
        self._pt_test_indices = []
        if self._pt_folds is not None and self._test_size == 1:
            self._pt_test_indices = self._test_fold(self._pt_fold)
        elif self._test_size > 0:
            if self._pt_split_stratify is None:
                if self._split_shuffle:
                    _, self._pt_test_indices = train_test_split(self._indices_before_testsplit, test_size=self._test_size, random_state=self._pt_split_random_state)
                    self._pt_test_indices = sorted(self._pt_test_indices)
                else:
                    test_size = int(self._test_size * len(self._indices_before_testsplit))
                    self._pt_test_indices = self._indices_before_testsplit[-test_size:]
            else:
                if len(self._pt_split_stratify) > 1:
                    splitter = MultilabelStratifiedShuffleSplit(n_splits=1, 
                                                       random_state=self._pt_split_random_state, 
                                                       test_size=self._test_size)
                else:
                    splitter = StratifiedShuffleSplit(n_splits=1, 
                                                       random_state=self._pt_split_random_state, 
                                                       test_size=self._test_size) 
                target = self.iloc[self._indices_before_testsplit][self._pt_split_stratify]
                _, self._pt_test_indices = next(splitter.split(target, target))
                self._pt_test_indices = sorted(self._pt_test_indices)
                    
        return self._pt_test_indices
    
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
            indices = np.array(indices)[ mask.astype(int) ]
        return indices

    @property
    def _valid_indices(self):
        try:
            if self._pt__locked_valid_indices is not None:
                return self._pt__locked_valid_indices
        except: pass
        try:
            if self._pt_valid_indices is not None:
                return self._pt_valid_indices
        except: pass
        try:
            del self._pt_train_indices
        except: pass
        self._pt_valid_indices = []
        if self._pt_folds is not None:
            self._pt_valid_indices = self._valid_fold(self._pt_fold)
        elif self._valid_size > 0:
            if self._test_size < 1:
                valid_size = self._valid_size / (1 - self._test_size)
            else:
                valid_size = self._valid_size
            if valid_size > 0:
                if self._pt_split_stratify is None:
                    if self._shuffle:
                        _, self._pt_valid_indices = train_test_split(self._indices_after_testsplit, test_size=valid_size, random_state=self._pt_split_random_state)
                        self._pt_valid_indices = sorted(self._pt_valid_indices)
                    else:
                        valid_size = int(valid_size * len(self._indices_before_testsplit))
                        self._pt_valid_indices = self._indices_after_testsplit[-valid_size:]                        
                else:
                    if len(self._pt_split_stratify) > 1:
                        splitter = MultilabelStratifiedShuffleSplit(n_splits=1, 
                                                           random_state=self._pt_split_random_state, 
                                                           test_size=valid_size)
                    else:
                        splitter = StratifiedShuffleSplit(n_splits=1, 
                                                           random_state=self._pt_split_random_state, 
                                                           test_size=valid_size) 
                    target = self.loc[self._indices_after_testsplit, self._pt_split_stratify]
                    _, self._pt_valid_indices = next(splitter.split(target, target))
                    self._pt_valid_indices = sorted(self._pt_valid_indices)
        elif self._pt_folds is not None:
            self._pt_valid_indices = self._valid_fold(self._pt_fold)
        return self._pt_valid_indices

    @property
    def _folds(self):
        try:
            if self._pt_created_folds is not None:
                return self._pt_created_folds
        except: pass
        assert self._pt_folds is not None and type(self._pt_folds) == int, 'You have to set split(folds) to an integer'
        assert self._pt_folds > 1, 'You have to set split(folds) to an integer > 1'
        self._pt_created_folds = []
        if 0 < self._test_size < 1:
            indices = self._indices_after_testsplit
        else:
            indices = self._indices_before_testsplit
        if self._pt_folds_stratify is None:
            target = self.iloc[indices]
            splitter = KFold(n_splits = self._pt_folds, shuffle=self._pt_folds_shuffle, random_state=self._pt_folds_random_state)
            for train_indices, valid_indices in splitter.split(target, target):
                self._pt_created_folds.append(sorted(indices[valid_indices]))
        else:
            if len(self._pt_folds_stratify) > 1:
                splitter = MultilabelStratifiedKFold(n_splits = self._pt_folds,
                                    shuffle=True,
                                    random_state=self._pt_folds_random_state)
            else:
                splitter = StratifiedKFold(n_splits = self._pt_folds,
                                    shuffle=True,
                                    random_state=self._pt_folds_random_state)
            target = self.loc[indices, self._pt_folds_stratify]
            for train_indices, valid_indices in splitter.split(target, target):
                self._pt_created_folds.append(sorted(indices[valid_indices]))
        return self._pt_created_folds
    
    def _test_fold(self, i):
        test_fold = (i + 1 + (i // self._pt_folds % (self._pt_folds - 1))) % self._pt_folds
        return self._folds[test_fold]
    
    def _valid_fold(self, i):
        return self._folds[i]
    
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
        r = copy.copy(self)
        
        i = i % r._pt_folds
        r._pt_fold = i
        if self._test_size == 1:  # choose a fold
            r._pt_test_indices = r._test_fold(i)
            r._pt_valid_indices = r._valid_fold(i)
            #r._pt_train_indices = np.hstack([ f for f in self._folds if f != r._pt_test_indices and f != r._pt_valid_indices ])
        else:
            r._pt_valid_indices = r._valid_fold(i)
            #r._pt_train_indices = np.hstack([ f for f in self._folds if f != r._pt_valid_indices ])
        return r
    
    def iterfolds(self):
        """
        Iterate over the folds for n-fold cross validation. 
        
        A first call will trigger rows to be assigned to the train, valid and test part, which are stored
        in place to reproduce the exact same split for consecutive calls.
        
        Yields:
            train, valid (DSet)
        """
        r = self.fold(0)
        for i in range(r._pt_folds):
            yield r.fold(i)
    
    def df_to_dataset(self, df):
        """
        Converts the given df to a DataSet using the pipeline of this DFrame.
        
        Arguments:
            df: DataFrame or DFrame
                to convert into a DataSet
        
        returns: Converts the given df to a DataSet.
        """
        #assert self.is_locked, 'You can only use a locked DFrame, to prevent inconsistencies in the transformation'
        return self.df_to_dset(df).to_dataset()
        
    def to_datasets(self, dataset=None):
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
        #assert self.is_locked, 'You can only use a locked DFrame, to prevent inconsistencies in the transformation'
        return self._dset(df, range(len(df)))
        
    def to_databunch(self, dataset=None, batch_size=32, valid_batch_size=None, 
                     num_workers=0, shuffle=True, pin_memory=False, balance=False, 
                     collate=None):
        """
        Prepare the data as a Databunch that contains dataloaders for the train, valid and test part.
        batch_size, num_workers, shuffle, pin_memory: see Databunch/Dataloader constructor.
        
        A first call will trigger rows to be assigned to the train, valid and test part, which are stored
        in place to reproduce the exact same split for consecutive calls.

        Returns: Databunch
        """
        return Databunch(self, *self.to_datasets(dataset=dataset), 
                         batch_size=batch_size, valid_batch_size=valid_batch_size, 
                         num_workers=num_workers, shuffle=shuffle, 
                         pin_memory=pin_memory, balance=balance, collate=collate)    

    def evaluate(self, *metrics):
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
        Prepares the train subset as a DSet and optionally trains transformation parameters, e.g. for image
        normalization or text tokenization. Transformations are only supported for PyTorch DataSets.
        
        A first call will trigger rows to be assigned to the train, valid and test part, which are stored
        in place to reproduce the exact same split for consecutive calls.
        
        Returns: DSet
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
        
        This effect is not inplace, but configured to a copy that is returned. 

        Arguments:
            weights: True or dict
                when set to True, the target values of the training set are 
                uniformely distributed,
                otherwise a dictionary can be passed that map target values to the 
                desired fraction of the training set (e.g. {0:0.4, 1:0.6}).
        
        Returns: DFrame
        """
        r = self._copy_with_indices()
        try:
            del r._pt__locked_train_indices
        except: pass
        r._pt_balance = weights
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
        r = self._copy_with_indices()
        r._pt_scale_columns = columns
        r._pt_scalertype = scalertype
        r._pt_scale_omit_interval = omit_interval
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
        r = self._copy_with_indices()
        r._pt_bias = True
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
        r = self.reset_indices()
        r._pt_split_random_state = random_state
        return r
        
    def reset_indices(self):
        """
        Clears the currently sampled split() and folds(), which is stored whenever data preparation
        is called. Therefore, resampling the split or fold on the next call to data preparation.
        However, this call will not reset a random_state, use reshuffle() for that.
        
        This effect is not inplace, but applied to a copy that is returned. 
        
        Returns: DFrame
        """
        r = copy.copy(self)
        try:
            del r._pt_train_indices
        except: pass
        try:
            del r._pt_valid_indices
        except: pass
        try:
            del r._pt_test_indices
        except: pass
        try:
            del r._pt_created_folds
        except: pass
        return r
        
    def split(self, valid_size=None, test_size=None, shuffle=None, random_state=None, stratify=None):
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
            stratify: str or [ str ] (None)
                apply stratified sampling. Per value for the given column, the rows are sampled. When a list
                of columns is given, multi-label stratification is applied.
            
        Returns: DFrame 
        """
        r = self.reset_indices()
        r._pt_valid_size = valid_size
        r._pt_test_size = test_size
        r._pt_split_shuffle = shuffle
        r._pt_split_random_state = random_state
        r._pt_split_stratify = [stratify] if type(stratify) == str else stratify
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
                set a random_state for reproducible results
            stratify: str or [ str ] (None)
                apply stratified sampling. Per value for the given column, the rows are sampled. When a list
                of columns is given, multi-label stratification is applied.
            test: bool (None)
                whether to use one fold as a test set. The default None is interpreted as True when
                split is not used. Often for automated n-fold cross validation studies, the validation set
                is used for early termination, and therefore you should use an out-of-sample
                test set that was not used for optimizing.
            
        Returns: copy of DFrame 
            schedules the data to be split in folds.
        """
        r = self.reset_indices()
        r._pt_folds = folds
        r._pt_folds_shuffle = shuffle
        r._pt_folds_random_state = random_state
        r._pt_folds_stratify = [stratify] if type(stratify) == str else stratify
        if test or (r._pt_test_size is None and test is None):
            r._pt_test_size = 1
        return r
        
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
        r = self._copy_with_indices()
        r._pt_polynomials = PolynomialFeatures(degree, include_bias=include_bias)
        return r
    
    def no_columny(self):
        """
        PipeTorch cannot currently handle a dataset without a target variable, 
        however, it can work by simply assigning one of the used input features
        as a target variable.

        This effect is not inplace, but configured to a copy that is returned. 

        Returns: copy of DFrame
        """
        r = self._copy_with_indices()
        r._pt_columny = [self._pt_columnx[0]]
        return r
    
    def columny(self, columns=None, vector=False):
        """
        Configures the generated target variable and shape.
        
        This effect is not inplace, but configured to a copy that is returned. 

        Arguments:
            columns: str or list of str (None)
                single column name or list of columns that is to be used as target column. 
                None: use the last column
            vector: bool (False)
                Some algorithms (e.g. knn) prefer y as a vector instead of an (n, 1) matrix. Setting 
                vector=True returns the target variable as a vector instead of an (n, 1) matrix. This
                can only be chosen when there is at most 1 target variable.
        
        Returns: DFrame 
        """
        
        r = self._copy_with_indices()
        if columns is not None:
            r._pt_columny = [columns] if type(columns) == str else columns
        assert r._pt_columny is None or len(r._pt_columny) == 1 or not vector, 'You cannot create target vector with multiple columns'
        r._pt_vectory = vector
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
        r = self._copy_with_indices()
        r._pt_category = columns
        r._pt_category_sort = sort
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
        r = self._copy_with_indices()
        r._pt_sequence_window = window
        r._pt_sequence_shift_y = shift_y
        return r
    
    def train_transforms(self, *transforms):
        """
        Configure a (list of) transformation function(s) that is called from the DataSet class to prepare the 
        train data.
        
        This effect is not inplace, but configured to a copy that is returned. 

        Arguments:
            *transforms: [ callable ]
                (list of) transformation function(s) that is called from the DataSet class to prepare the data.

        Returns: DFrame
        """
        r = self._copy_with_indices()
        r._pt_train_transforms = transforms
        return r
    
    def transforms(self, *transforms):
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
        r = self._copy_with_indices()
        r._pt_transforms = transforms
        return r

    def _train_transformation_parameters(self, train_dset):
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
            A list of transformations that is applied to the generated DSet 
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
    
class DFrame(pd.DataFrame, _DFrame):
    _metadata = _DFrame._metadata
    _internal_names = _DFrame._internal_names
    _internal_names_set = _DFrame._internal_names_set

    def __init__(self, data, *args, **kwargs):
        super().__init__(data, *args, **kwargs)
        _DFrame.__init__(self, data)
        
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
    def read_from_kaggle(cls, dataset, train=None, test=None, shared=False, force=False, **kwargs):
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
        train = read_from_kaggle(dataset, filename=train, shared=shared, force=force, **kwargs)
        if test is not None:
            test = read_from_kaggle(dataset, filename=test, **kwargs)
            return cls.from_train_test(train, test)
        return cls(train)
           
    @classmethod
    def read_from_kaggle_competition(cls, dataset, train=None, test=None, shared=False, force=False, **kwargs):
        train = read_from_kaggle_competition(dataset, filename=train, shared=shared, force=force, **kwargs)
        if test is not None:
            test = read_from_kaggle_competition(dataset, filename=test, **kwargs)
            return cls.from_train_test(train, test)
        return cls(train)

    @classmethod
    def read_csv(cls, url, filename=None, path=None, save=False, **kwargs):
        """
        Reads a .csv file from cache or url. The place to store the file is indicated by path / filename
        and when a delimiter is used, this is also used to save the file so that the original delimiter is kept.
        The file is only downloaded using the url if it does not exsists on the filing system. If the file is
        downloaded and save=True, it is also stored for future use.

        Arguments:
            url: str
                the url to download or a full path pointing to a .csv file
            filename: str (None)
                the filename to store the downloaded file under. If None, the filename is extracted from the url.
            path: str (None)
                the path in which the file is stored. If None, it will first check the ~/.pipetorch (for sharing
                dataset between users) and then ~/.pipetorchuser (for user specific caching of datasets).
            save: bool (False)
                whether to save a downloaded .csv
            **kwargs:
                additional parameters passed to pd.read_csv. For example, when a multichar delimiter is used
                you will have to set engine='python'.

        Returns: DFrame
        """
        return cls(read_csv(url, filename=filename, path=path, save=save, **kwargs))
   
    @classmethod
    def read_from_package(cls, package, filename, **kwargs):
        return cls(read_from_package(package, filename))
    
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
