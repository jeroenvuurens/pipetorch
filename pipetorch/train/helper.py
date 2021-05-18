import torch
import os
import matplotlib
from matplotlib import pyplot as plt
import matplotlib.patheffects as PathEffects
from IPython.core import pylabtools as pt
from pathlib2 import Path
from sklearn.manifold import TSNE
import seaborn as sns
import numpy as np
import sys
from IPython import get_ipython
ipython = get_ipython()
back2gui = { b:g for g, b in pt.backends.items() }

class plt_gui:
    def __init__(self, gui):
        self.gui = gui

    def __enter__(self):
        backend = matplotlib.get_backend()
        self.old_gui = back2gui[backend]
        ipython.magic('matplotlib ' + self.gui)

    def __exit__(self, *args):
        ipython.magic('matplotlib ' + self.old_gui)

class plt_inline(plt_gui):
    def __init__(self):
        super().__init__('inline')

class plt_notebook(plt_gui):
    def __init__(self):
        super().__init__('notebook')

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
        self.figure = plt.figure()
        self.ax = self.figure.add_subplot(111)
        self.figure.show()
        self.xlabel = xlabel
        self.ylabel = ylabel
        self.xscale = xscale
        self.yscale = yscale

    def __enter__(self):
        plt.ion()
        return self

    def __exit__(self, *args):
        plt.ioff()

    def set_ylim(self, y):
        y = np.array(y)
        while True:
            mean_y = np.mean(y)
            sd_y = np.std(y)
            keep = (y >= mean_y - 4 * sd_y) & (y <= mean_y + 4 * sd_y)
            if sum(keep) == len(y):
                break
            y = y[keep]
        if min(y) < max(y):
            self.ax.set_ylim(max(y) - (max(y) - min(y)) * 1.1, min(y) + (max(y) - min(y)))

    def set_ylim_multi(self, yy):
        min_y = None
        max_y = None
        for y in yy.values():
            y = np.array(y)
            while True:
                mean_y = np.mean(y)
                sd_y = np.std(y)
                keep = (y >= mean_y - 3 * sd_y) & (y <= mean_y + 3 * sd_y)
                if sum(keep) == len(y):
                    break
                y = y[keep]
            if min_y is not None:
                min_y = min(min_y, min(y))
                max_y = max(max_y, max(y))
            else:
                min_y = min(y)
                max_y = max(y)
        if min_y < max_y:
            self.ax.set_ylim(max_y - (max_y - min_y) * 1.05, min_y + (max_y - min_y)*1.05)

    def replot(self, x, y):
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

    def multiplot(self, x, yy):
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

