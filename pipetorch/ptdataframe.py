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
from .evaluate import Evaluator
from .databunch import Databunch
from .ptdataset import PTDataSet

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

def read_excel(path, alternativesource=None, sep=None, delimiter=None, **kwargs):
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

def read_csv(path, alternativesource=None, sep=None, delimiter=None, **kwargs):
    if sep:
        kwargs['sep'] = sep
    elif delimiter:
        kwargs['delimiter'] = delimiter
    filename = get_filename(path)
    if (Path.home() / '.pipetorchuser' / filename).is_file():
        #print(str(Path.home() / '.pipetorchuser' / filename))
        return PTDataFrame.read_csv(Path.home() / '.pipetorchuser' / filename, **kwargs)
    if (Path.home() / '.pipetorch' / filename).is_file():
        #print(str(Path.home() / '.pipetorch' / filename))
        return PTDataFrame.read_csv(Path.home() / '.pipetorch' / filename, **kwargs)
    if alternativesource:
        df = PTDataFrame(alternativesource())
    else:
        print('Downloading new file ' + path)
        df = pd.read_csv(path, **kwargs)
    (Path.home() / '.pipetorchuser').mkdir(exist_ok=True)
    df.to_csv(Path.home() / '.pipetorchuser' / filename, index=False)
    return df

class PTDataFrame(pd.DataFrame):
    _metadata = ['_pt_scale_columns', '_pt_scale_omit_interval', '_pt_scalertype', '_pt_train_indices', '_pt_train_indices_unbalanced', '_pt_valid_indices', '_pt_test_indices', '_pt_columny', '_pt_transposey', '_pt_bias', '_pt_polynomials']

    _internal_names = pd.DataFrame._internal_names + ['_pt__scale_columns', '_pt__train', '_pt__valid', '_pt__test', '_pt__full', '_pt__train_unbalanced', '_pt__scalerx', '_pt__scalery', '_pt__train_x', '_pt__train_y', '_pt__valid_x', '_pt__valid_y']
    _internal_names_set = set(_internal_names)

    @classmethod
    def read_csv(cls, path, **kwargs):
        df = pd.read_csv(path, **kwargs)
        return cls(df)

    def __init__(self, data, **kwargs):
        super().__init__(data, **kwargs)
        if type(data) != self._constructor:
            for m in self._metadata:
                self.__setattr__(m, None)
    
    @property
    def _constructor(self):
        return PTDataFrame

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
            
    @property
    def _columnx(self):
        return [ c for c in self.columns if c not in self._columny ]
   
    def _scalerx(self):
        try:
            if self._pt__scalerx is not None:
                return self._pt__scalerx
        except: pass
        X = self.train._x_polynomials
        if self._pt_scale_columns == True or self._pt_scale_columns == 'x_only':         
            self._pt__scalerx = tuple( self._create_scaler(self._pt_scalertype, X[:, i:i+1]) if (X[:,i].min() < self._pt_scale_omit_interval[0] or X[:,i].max() > self._pt_scale_omit_interval[1]) else None for i in range(X.shape[1]) )    
        elif self._pt_scale_columns == False or self._pt_scale_columns is None or len(self._pt_scale_columns) == 0:
            self._pt__scalerx = tuple( [ None ] * X.shape[1] )
        else:
            self._pt__scalerx = tuple( self._create_scaler(self._pt_scalertype, X[:, i:i+1]) if c in self._pt_scale_columns else None for i, c in enumerate(self._columnx) )
        return self._pt__scalerx
        
    def _scalery(self):
        try:
            if self._pt__scalery is not None:
                return self._pt__scalery
        except: pass
        y = self.train._y_numpy
        if self._pt_scale_columns == True:         
            self._pt__scalery = tuple( self._create_scaler(self._pt_scalertype, y[:, i:i+1]) if (y[:,i].min() < self._pt_scale_omit_interval[0] or y[:,i].max() > self._pt_scale_omit_interval[1]) else None for i in range(y.shape[1]) )    
        elif self._pt_scale_columns == False or self._pt_scale_columns is None or len(self._pt_scale_columns) == 0 or self._pt_scale_columns == 'x_only':
            self._pt__scalery = tuple( [None] * y.shape[1] )
        else:
            self._pt__scalery = tuple( self._create_scaler(self._pt_scalertype, y[:, i:i+1]) if c in self._pt_scale_columns else None for i, c in enumerate(self._columny) )
        return self._pt__scalery
        
    def columny(self, columns=None, transpose=True):
        r = copy.deepcopy(self)
        if columns is not None:
            r._pt_columny = [columns] if type(columns) == str else columns
        if transpose is not None:
            r._pt_transposey = transpose
        return r
   
    @property
    def _indices(self):
        return np.where(self.notnull().all(1))[0]

    @property
    def _train_indices(self):
        if self._pt_train_indices is None:
            return self._indices
        return self._pt_train_indices

    @property
    def _valid_indices(self):
        if self._pt_valid_indices is None:
            return [ ]
        return self._pt_valid_indices

    @property
    def _test_indices(self):
        if self._pt_test_indices is None:
            return [ ]
        return self._pt_test_indices

    @property
    def _train_indices_unbalanced(self):
        if self._pt_train_indices_unbalanced is None:
            return self._train_indices
        return self._pt_train_indices_unbalanced
 
    def to_dataset(self):
        """
        returns: a list with a train, valid and test DataSet. Every DataSet contains an X and y, where the 
        input data matrix X contains all columns but the last, and the target y contains the last column
        columns: list of columns to convert, the last column is always the target. default=None means all columns.
        """
        import torch
        from torch.utils.data import TensorDataset, DataLoader
        p = [ TensorDataset(torch.from_numpy(self.train.X), torch.from_numpy(self.train.y)) ]
        p.append( TensorDataset(torch.from_numpy(self.valid.X), torch.from_numpy(self.valid.y)) )
        p.append( TensorDataset(torch.from_numpy(self.test.X), torch.from_numpy(self.test.y)) )
        return p
    
    def to_databunch(self, batch_size=32, num_workers=0, shuffle=True, pin_memory=False, balance=False):
        """
        returns: a Databunch that contains dataloaders for the train, valid and test part.
        batch_size, num_workers, shuffle, pin_memory: see Databunch/Dataloader constructor
        """
        return Databunch(*self.to_dataset(), batch_size=batch_size, num_workers=num_workers, shuffle=shuffle, pin_memory=pin_memory, scaler=self, balance=balance)    

    def evaluate(self, *metrics):
        #assert len(metrics) > 0, 'You need to provide at least one metric for the evaluation'
        return Evaluator(self, metrics)
    
    def _ptdataset(self, data):
        r = PTDataSet.from_ptdataframe(data, self)
        r._pt_scalerx = self._scalerx
        r._pt_scalery = self._scalery
        r._pt_columny = self._columny
        r._pt_transposey = self._transposey
        r._pt_polynomials = self._pt_polynomials
        r._pt_bias = self._pt_bias
        return r

    def _ptdataframe(self, data):
        r = self._constructor(data)
        r._pt__scalerx = self._scalerx()
        r._pt__scalery = self._scalery()
        r._pt_columny = self._columny
        r._pt_transposey = self._transposey
        r._pt_polynomials = self._pt_polynomials
        r._pt_bias = self._pt_bias
        return r  
    
    @classmethod
    def _split_indices(self, indices, split):
        length = len(indices)
        try:
            train_length = int((1 - sum(split))* length)
            train_valid_length = int((1 - split[1])* length)
            assert train_length >= 0, 'Non positive size of the training set, provide fractions for valid/test part, e.g. (0.2, 0.3)'
            assert train_valid_length >= train_length, 'Non positive size of the validation set, provide fraction for valid part, bigger than 0, e.g. e.g. (0.2, 0.3)'
            assert length >= train_valid_length, 'Negative fraction of the test set, provide fractions for valid/test part, e.g. (0.2, 0.3)'
            train_indices = indices[:train_length]
            valid_indices = indices[train_length:train_valid_length]
            test_indices = indices[train_valid_length:]
        except:
            train_length = int((1 - split)* length)
            assert train_length >= 0, 'Non positive size of the training set, provide fraction for valid part, smaller than 1, e.g. 0.2'
            assert train_length <= length, 'Non positive size of the validation set, provide fraction for valid part, bigger than 0, e.g. 0.2'
            train_indices = indices[:train_length]
            valid_indices = indices[train_length:]
            test_indices = []
        return train_indices, valid_indices, test_indices

    def split(self, split=0.2, shuffle=True, random_state=None):
        assert len(self._train_indices_unbalanced) == len(self._train_indices), "Split the DataFrame before balancing!"
        r = copy.deepcopy(self)
        indices = r._indices
        if shuffle:
            if random_state is not None:
                np.random.seed(random_state)
            np.random.shuffle(indices)
            if random_state is not None:
                t = 1000 * time.time() # current time in milliseconds
                np.random.seed(int(t) % 2**32)
        r._pt_train_indices, r._pt_valid_indices, r._pt_test_indices = r._split_indices(indices, split)
        r._pt_train_indices_unbalanced = r._pt_train_indices
        return r
    
    def resample(self, n=True):
        r = copy.copy(self)
        if n == True:
            n = len(r._pt_train_indices_unbalanced)
        if n < 1:
            n = n * len(r._pt_train_indices_unbalanced)
        if n > 0:
            r._pt_train_indices = resample(r._pt_train_indices_unbalanced, n_samples = int(n))
        return r
    
    def polynomials(self, degree):
        #assert not self._check_attr('_bias'), 'You should not add a bias before polynomials, rather use include_bias=True'
        assert type(self._pt_scale_columns) != list or len(self._pt_scale_columns) == 0, 'You cannot combine polynomials with column specific scaling'
        r = copy.copy(self)
        r._pt_polynomials = PolynomialFeatures(degree, include_bias=False)
        try:
            del r._pt_scalerx
            del r._pt_scalery
        except: pass
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
    
    @property
    def train_unbalanced(self):
        try:
            return self._pt__train_unbalanced
        except:
            self._pt__train_unbalanced = self._ptdataset(self.iloc[self._train_indices_unbalanced])
        return self._pt__train_unbalanced

    @property
    def full(self):
        try:
            return self._pt__full
        except:
            self._pt__full = self._ptdataset(self.iloc[np.concatenate([self._train_indices, self._valid_indices])])
        return self._pt__full

    @property
    def train(self):
        try:
            return self._pt__train
        except:
            self._pt__train = self._ptdataset(self.iloc[self._train_indices])
        return self._pt__train
    
    @property
    def valid(self):
        try:
            return self._pt__valid
        except:
            self._pt__valid = self._ptdataset(self.iloc[self._valid_indices])
        return self._pt__valid

    @property
    def test(self):
        try:
            return self._pt__test
        except:
            self._pt__test = self._ptdataset(self.iloc[self._test_indices])
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
        try:
            del r._pt_scalerx
            del r._pt_scalery
        except: pass
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
        return pd.DataFrame(self._inverse_transform(to_numpy(y), self._scalery(), self._columny))

    def inverse_transform_X(self, X):
        if self._pt_bias:
            X = X[:, 1:]
        if self._pt_polynomials is not None:
            X = X[:, :len(self._columnx)]
        return self._inverse_transform(to_numpy(X), self._scalerx()[:len(self._columnx)], self._columnx)

    def _inverse_transform(self, data, scalerlist, columns):
        data = to_numpy(data)
        if scalerlist is not None:
            data = [ data[:, i:i+1] if scaler is None else scaler.inverse_transform(data[:,i:i+1]) for i, scaler in enumerate(scalerlist) ]
        series = [ pd.Series(x.reshape(-1), name=c) for x, c in zip(data, columns)]
        return pd.concat(series, axis=1)

    def inverse_transform(self, X, y):
        y = self.inverse_transform_y(y)
        X = self.inverse_transform_X(X)
        return self._ptdataset(pd.concat([X, y], axis=1))
    
    def balance(self, weights=None):
        if weights is None:
            return self._balance_y_equal()
        r = copy.deepcopy(self)
        y = r.train[r._columny]
        indices = {l:np.where(y==l)[0] for l in np.unique(y)}
        classlengths = {l:len(i) for l,i in indices}
        n = max([ int(math.ceil(classlength[c] / w)) for c, w in weights.items() ])
        mask = np.hstack([np.random.choice(i, n*weights[l]-classlength[l], replace=True) for l, i in indices.items()])
        indices = np.hstack([mask, range(len(y))])
        r._pt_train_indices = r._train_indices_unbalanced[indices]
        return r
    
    def _balance_y_equal(self):
        r = copy.deepcopy(self)
        y = r.train[r._columny]
        indices = [np.where(y==l)[0] for l in np.unique(y)]
        classlengths = [len(i) for i in indices]
        n = max(classlengths)
        mask = np.hstack([np.random.choice(i, n-l, replace=True) for l,i in zip(classlengths, indices)])
        indices = np.hstack([mask, range(len(y))])
        r._pt_train_indices = r._train_indices_unbalanced[indices]
        return r
    
    def plot_boundary(self, predict):
        self.evaluate().plot_boundary(predict)
