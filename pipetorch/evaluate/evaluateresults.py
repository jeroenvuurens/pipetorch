import numpy as np
import pandas as pd
import copy
import itertools
import matplotlib.pyplot as plt

class EvaluatorResults(pd.DataFrame):
    _metadata = ['evaluator', '_phase', 'df']
    
    @classmethod
    def from_evaluator(cls, evaluator):
        r = cls()
        r.evaluator = evaluator
        r.df = evaluator.df
        return r
    
    @property
    def _constructor(self):
        return EvaluatorResults

    def _copy_meta(self, df):
        df = self._constructor(df)
        for m in self._metadata:
            setattr(df, m, getattr(self, m))
        return df
    
    def __finalize__(self, other, method=None, **kwargs):
        for name in self._metadata:
            object.__setattr__(self, name, getattr(other, name, None))
        return self
    
    def __repr__(self):
        r = self.drop( columns = [ c for c in self.columns if c[0] == '_' ] )
        return super(EvaluatorResults, r).__repr__()
    
    def _add(self, row):
        for c in row.columns:
            if c not in self.columns:
                self.insert(len(self.columns), c, np.NaN)
        r = self.append(row, sort=True, ignore_index=True)
        r.evaluator = self.evaluator
        return r

    @property
    def train(self):
        r = self[self.phase == 'train']
        r._phase = 'train'
        return r
    
    @property
    def valid(self):
        r = self[self.phase == 'valid']
        r._phase = 'valid'
        return r
    
    @property
    def test(self):
        r = self[self.phase == 'test']
        r.phase = 'test'
        return r

    def _plot_predict(self, pltfunction, label, x=None, xlabel = None, ylabel = None, title=None, interpolate=0, df=None, **kwargs):
        marker = itertools.cycle((',', '+', '.', 'o', '*')) 
        for _, row in self.iterrows():
            y = row['_predict']
            try:
                l = label(row)
            except:
                l = row[label]
            self.evaluator._plot(pltfunction, x=x, y=y, xlabel=xlabel, ylabel=ylabel, title=title, marker=next(marker), interpolate=interpolate, df=df, label=l, **kwargs)
        plt.legend()
   
    def _select(self, select=None):
        if select is None:
            s = self.results
        elif type(select) is pd.core.series.Series:
            s = self.results[select]
        elif type(select) is EvaluatorResults:
            s = select
        elif type(select) is str:
            s = self.results[self.results.phase == select]
        else:
            raise ValueError('Unknown type passed for select')
        return s

    def _plot(self, pltfunction, x, y=None, xlabel = None, ylabel = None, title=None, label=None, loc='upper right', **kwargs):
        f = _figure(self, x=x, y=y, xlabel=xlabel, ylabel=ylabel, title=title)
        if label is not None:
            kwargs['label'] = label
        pltfunction(f.graph_x, f.graph_y, **kwargs)      
        if 'label' in kwargs:
            plt.legend(loc=loc)
        
    def line(self, x, y=None, xlabel = None, ylabel = None, title=None, **kwargs):
        self._plot(plt.plot, x, y=y, xlabel=xlabel, ylabel=ylabel, title=title, **kwargs)
    
    def scatter(self, x, y=None, xlabel = None, ylabel = None, title=None, **kwargs):
        self._plot(plt.scatter, x, y=y, xlabel=xlabel, ylabel=ylabel, title=title, **kwargs)
        
    def line_metric(self, x, series='phase', y=None, xlabel = None, ylabel = None, title=None, label_prefix='', **kwargs):
        self._evaluator.line_metric(x, series=series, select=self, y=y, xlabel=xlabel, ylabel=ylabel, title=title, label_prefix=label_prefix, **kwargs)

    def scatter_metric(self, x, series='phase', y=None, xlabel = None, ylabel = None, title=None, label_prefix='', **kwargs):
        self._evaluator.scatter_metric(x, series=series, select=self, y=y, xlabel=xlabel, ylabel=ylabel, title=title, label_prefix=label_prefix, **kwargs)

class _figure:
    def __init__(self, results, x=None, y=None, xlabel = None, ylabel = None, title = None ):
        self.evaluator = results.evaluator
        self.results = results
        self.x = x
        self.y = y
        self.xlabel = xlabel or self.x
        self.ylabel = ylabel or self.y
        if title is not None:
            plt.title(title)
        plt.ylabel(self.ylabel) 
        plt.xlabel(self.xlabel)
        self.results = self.results.sort_values(by=self.x)
        gx = [ row[self.x] for _, row in self.results.iterrows() ]
        gy = [ row[self.y] for _, row in self.results.iterrows() ]

    @property
    def x(self):
        return self._x
    
    @x.setter
    def x(self, value):
        self._x = value or self.df._columnx[0]
        
    @property
    def y(self):
        return self._y
    
    @y.setter
    def y(self, value):
        self._y = value or self.evaluator.metrics[0].__name__
        
    @property
    def xlabel(self):
        return self._xlabel
    
    @xlabel.setter
    def xlabel(self, value):
        self._xlabel = value or self.x

    @property
    def ylabel(self):
        return self._ylabel
    
    @ylabel.setter
    def ylabel(self, value):
        self._ylabel = value or self.y

    @property
    def graph_x(self):
        return [ row[self.x] for _, row in self.results.iterrows() ]

    @property
    def graph_y(self):
        return [ row[self.y] for _, row in self.results.iterrows() ]
