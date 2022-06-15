import torch
import os
import matplotlib
from matplotlib import pyplot as plt
import matplotlib.patheffects as PathEffects
# from IPython.core import pylabtools as pt
from pathlib import Path
from sklearn.manifold import TSNE
import seaborn as sns
import numpy as np
import sys
import math
from tqdm.notebook import tqdm

def getsizeof(o, ids=set()):
    d = deep_getsizeof
    if id(o) in ids:
        return 0

    r = sys.getsizeof(o)
    ids.add(id(o))

    if isinstance(o, str) or isinstance(0, unicode):
        return r

    if isinstance(o, Mapping):
        return r + sum(d(k, ids) + d(v, ids) for k, v in o.iteritems())

    if isinstance(o, Container):
        return r + sum(d(x, ids) for x in o)

    return r

class Plot:
    def __init__(self, xlabel=None, ylabel='Loss', xscale=None, yscale='log', **kwargs):
        self.xlabel = xlabel
        self.ylabel = ylabel
        self.xscale = xscale
        self.yscale = yscale
        self.interactive = matplotlib.get_backend() == 'nbAgg'
        if self.interactive:
            self.figure = plt.figure()
            self.ax = self.figure.gca()
            self.figure.show()

    def __enter__(self):
        if self.interactive:
            plt.ion()
        return self

    def __exit__(self, *args):
        if self.interactive:
            plt.ioff()
        else:
            self.figure = plt.figure()
            self.ax = self.figure.gca()
            if self.xlabel:
                self.ax.set_xlabel(self.xlabel)
            if self.ylabel:
                self.ax.set_ylabel(self.ylabel)
            if self.xscale:
                self.ax.set_xscale(self.xscale)
            if self.yscale:
                self.ax.set_yscale(self.yscale)
            try:
                self.ax.plot( self.x, self.y)
                self.set_ylim(self.y)
            except:
                try:
                    for name, y in self.yy.items():
                        self.ax.plot( self.x, y, label=str(name))
                    self.set_ylim_multi(self.yy)
                except: pass
            self.figure.show()

    def set_ylim(self, y):
        y = np.array(y)
        min_y = min(y)
        first_y = y[0]
        max_y = min(max(y), 4 * (first_y - min_y) + min_y)
        if min_y < max_y:
            self.ylim = max(0.1, max_y - (max_y - min_y) * 1.1), min_y + (max_y - min_y)
            self.ax.set_ylim(self.ylim)

    def set_ylim_multi(self, yy):
        min_y = min([ min(y) for y in yy.values() ])
        first_y = max([y[0] for y in yy.values() ])
        max_y = max([max(y) for y in yy.values() ])
        max_y = min(max_y, 4 * (first_y - min_y) + min_y)
        if min_y < max_y:
            self.ylim = max(0.1, max_y - (max_y - min_y) * 1.1), min_y + (max_y - min_y)
            self.ax.set_ylim(self.ylim)

    def replot(self, x, y):
        if self.interactive:
            self.ax.clear()
            if self.xlabel:
                self.ax.set_xlabel(self.xlabel)
            if self.ylabel:
                self.ax.set_ylabel(self.ylabel)
            if self.xscale:
                self.ax.set_xscale(self.xscale)
            if self.yscale:
                self.ax.set_yscale(self.yscale)
            self.set_ylim(y)
            self.ax.plot( x, y)
            plt.show()
            self.figure.canvas.draw()
        else:
            self.x = x
            self.y = y

    def multiplot(self, x, yy):
        if self.interactive:
            self.ax.clear()
            if self.xlabel:
                self.ax.set_xlabel(self.xlabel)
            if self.ylabel:
                self.ax.set_ylabel(self.ylabel)
            if self.xscale:
                self.ax.set_xscale(self.xscale)
            if self.yscale:
                self.ax.set_yscale(self.yscale)
            self.set_ylim_multi(yy)
            for name, y in yy.items():
                self.ax.plot( x, y, label=str(name))
            self.ax.legend()
            self.figure.canvas.draw()
        else:
            self.x = x
            self.yy = yy

class tqdm_trainer(tqdm):
    """
    Extends tqdm for the PipeTorch Trainer. Typically, this is called with the number of epochs, cycle and 
    dataloaders that are used for training and 
    
    Arguments:
        epochs: int
            the number of epochs to train
        cycle: int
            the cycle configures after how many epochs the validation is run and reported
        train_dl: PyTorch DataLoader
            the dataloader that is used for training
        valid_dl: PyTorch DataLoader
            the dataloader that is used for validation
    """
    def __init__(self, epochs, cycle, train_dl, valid_dl, test_dl=None, folds=1, silent=False, iterable=None, desc='Total', total=None, leave=False):
        self.train_dl = train_dl
        self.valid_dl = valid_dl
        self.test_dl = test_dl
        self.epochs = epochs
        self.cycle = cycle
        self.folds = folds
        self.reportsleft = self._reports(epochs, cycle)
        self.batches_per_fold = self._batches(epochs, self.reportsleft, train_dl, valid_dl, test_dl)
        self.silent = silent
        if not silent:
            super().__init__(desc=desc, total=self.folds * self.batches_per_fold, leave=leave)

    @property
    def batches_left(self):
        try:
            return self._batches_left
        except:
            self._batches_left = self.batches_per_fold
            return self._batches_left
        
    @batches_left.setter
    def batches_left(self, value):
        self._batches_left = value
            
    def update(self, size):
        self.batches_left -= size
        if not self.silent:
            super().update(size)
    
    def finish_fold(self):
        self.total -= self.batches_left
        del self._batches_left
        if not self.silent:
            self.refresh()
    
    def close(self):
        if not self.silent:
            super().close()
            
    @classmethod
    def _reports(self, epochs, cycle):
        """
        Arguments:
            epochs: int
                the number of epochs to train
            cycle: int
                the cycle configures after how many epochs the validation is run and reported
           
        Returns: int - the number of times validation is run and reported
        """
        return math.ceil(epochs / cycle)
    
    @classmethod
    def _batches(self, epochs, reports, train_dl, valid_dl, test_dl=None):
        """
        Arguments:
            epochs: int
                the number of epochs to train
            reports: int
                the number of times validation is run and reported
            train_dl: PyTorch DataLoader
                the dataloader that is used for training
            valid_dl: PyTorch DataLoader
                the dataloader that is used for validation
            test_dl: PyTorch DataLoader
                the dataloader that is used for testing

        Returns: int - the total number of batches that is processed
        """
        
        batches = len(train_dl) * train_dl.batch_size * epochs
        batches += len(valid_dl) * valid_dl.batch_size * reports
        try:
            batches += len(test_dl) * test_dl.batch_size * reports
        except: pass
        return batches

def to_numpy(arr):
    if type(arr) is torch.Tensor:
        if arr.device.type == 'cuda':
            return arr.data.cpu().numpy()
        else:
            return arr.data.numpy()
    return arr

def plot_histories(metric, history, train=True, valid=True, **kwargs):
    plt.figure(**kwargs)
    for label, t in history.items():
        h = t.history
        x = [ epoch['epoch'] for epoch in h.epochs['train'] ]
        if train:
            plt.plot(x, h.train(metric), label=f'train_{label}')
        if valid:
            plt.plot(x, h.valid(metric), label=f'valid_{label}')
    plt.ylabel(metric.__name__)
    plt.xlabel("epochs")
    plt.legend()
    plt.show()

def create_path(p, mode=0o777):
    path = Path(p)
    os.makedirs(path, mode, exist_ok=True)
    return path

def scatter(x, colors):
    num_classes = len(np.unique(colors))
    palette = np.array(sns.color_palette("hls", num_classes))

    f = plt.figure(figsize=(8, 8))
    ax = plt.subplot(aspect='equal')
    sc = ax.scatter(x[:,0], x[:,1], lw=0, s=40, c=palette[colors.astype(np.int)])
    plt.xlim(-25, 25)
    plt.ylim(-25, 25)
    ax.axis('off')
    ax.axis('tight')

    txts = []

    for i in range(num_classes):
        xtext, ytext = np.median(x[colors == i, :], axis=0)
        txt = ax.text(xtext, ytext, str(i), fontsize=24)
        txt.set_path_effects([
            PathEffects.Stroke(linewidth=5, foreground="w"),
            PathEffects.Normal()])
        txts.append(txt)

    #return f, ax, sc, txts

def to_numpy1(a):
    try:
        a = a.detach()
    except: pass
    try:
        a = a.numpy()
    except: pass
    return a

def draw_regression(x, y_true, y_pred):
    f = plt.figure(figsize=(8, 8))
    x, y_true, y_pred = [to_numpy(a) for a in (x, y_true, y_pred)]
    plt.scatter(x, y_true)
    indices = np.argsort(x)
    plt.plot(x[indices], y_pred[indices])

def line_predict(x, y_true, y_pred):
    draw_regression(x, y_true, y_pred)

def scatter(x, y):
    f = plt.figure(figsize=(8, 8))
    x, y = [to_numpy(a) for a in (x, y)]
    plt.scatter(x, y)

def range3(start, end):
    while start < end:
        yield start
        yield start * 3
        start *= 10

def plot_tsne(X, y, random_state=0):
    t = TSNE(random_state=random_state).fit_transform(X)
    scatter(t, y)

def trace_warnings():
    import traceback
    import warnings
    import sys

    def warn_with_traceback(message, category, filename, lineno, file=None, line=None):
        log = file if hasattr(file,'write') else sys.stderr
        traceback.print_stack(file=log)
        log.write(warnings.formatwarning(message, category, filename, lineno, line))
    warnings.showwarning = warn_with_traceback

def expand_features(df, target, *features):
    if len(features) == 0:
        return [c for c in df.columns if c != target]
    else:
        return [c for c in features if c != target]

def read_csv(filename, nrows=100, drop=None, columns=None, dtype=dict(), intcols=[], **kwargs):
    df = pd.read_csv(filename, nrows=nrows, engine='python', **kwargs)
    if drop:
        df = df.drop(columns=drop)
    elif columns:
        df = df[columns]
    float_cols = [c for c in df if df[c].dtype.kind == "f" or df[c].dtype.kind == "i"]
    float32_cols = {c:np.float32 for c in float_cols}
    float32_cols.update({ c:np.int64 for c in intcols })
    float32_cols.update(dtype)
    df = pd.read_csv(filename, dtype=float32_cols, engine='python', low_memory=False, **kwargs)
    if drop:
        df = df.drop(columns=drop)
    elif columns:
        df = df[columns]
    return df

class nonondict(dict):
    """
    A dict that does not store None values, which is used to keep a
    dict of parameters for function calls, in which setting to None
    does not override the default setting.
    """
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.update(*args, **kwargs)
        
    def __setitem__(self, key, value):
        if value is None:
            try:
                del self[key]
            except: pass
        else:
            super().__setitem__(key, value)

    def setifnone(self, key, value):
        """
        Set a key to a value, only if that key does not yet exists.
        Since None values are not added, this also applies to keys
        that are previously set to None.
        
        Arguments:
            key: str
            value: any
        """
        if key not in self:
            self[key] = value
            
    def update(self, *args, **kwargs):
        for k, v in dict(*args, **kwargs).items():
            self[k] = v
