import numpy as np
import pandas as pd
import copy
import itertools
import matplotlib.pyplot as plt

class EvaluatorResults(pd.DataFrame):
    _metadata = ['_evaluator', '_phase']
    
    @classmethod
    def from_evaluator(cls, evaluator):
        r = cls()
        r._evaluator = evaluator
        return r
    
    @property
    def _constructor(self):
        return EvaluatorResults
    
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
        r._evaluator = self._evaluator
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

    def _figure(self, x, y=None, xlabel = None, ylabel = None, title = None, label=None):
        if y is None:
            y = self._evaluator.metrics[0].__name__
        if not xlabel:
            if type(x) == str:
                xlabel=x
            else:
                xlabel=df._columnx[0]
        if not ylabel:
            if type(y) == str:
                ylabel=y
            else:
                ylabel=df._columny[0]
        if label is None:
            try:
                label = self._phase
            except: None
        if title is not None:
            plt.title(title)
        if type(ylabel) == str:
            plt.ylabel(ylabel) 
        if type(xlabel) == str:
            plt.xlabel(xlabel)
        sort = self.sort_values(by=x)
        gx = [ row[x] for _, row in sort.iterrows() ]
        gy = [ row[y] for _, row in sort.iterrows() ]
        return gx, gy, label
        
    def _plot(self, pltfunction, x, y=None, xlabel = None, ylabel = None, title=None, label=None, **kwargs):
        gx, gy, label = self._figure(x, y=y, xlabel=xlabel, ylabel=ylabel, title=title, label=label)
        if label is not None:
            kwargs['label'] = label
        pltfunction(gx, gy, **kwargs)      

    def line_predict(self, label, x=None, xlabel = None, ylabel = None, title=None, interpolate=0, df=None, **kwargs):
        self._plot_results(plt.plot, label, x=x, xlabel=xlabel, ylabel=ylabel, title=title, interpolate=interpolate, df=df, **kwargs)
    
    def scatter_predict(self, label, x=None, xlabel = None, ylabel = None, title=None, interpolate=0, df=None, **kwargs):
        self._plot_results(plt.scatter, label, x=x, xlabel=xlabel, ylabel=ylabel, title=title, interpolate=interpolate, df=df, **kwargs) 
        
    def line(self, x, y=None, xlabel = None, ylabel = None, title=None, **kwargs):
        self._plot(plt.plot, x, y=y, xlabel=xlabel, ylabel=ylabel, title=title, **kwargs)
    
    def scatter(self, x, y=None, xlabel = None, ylabel = None, title=None, **kwargs):
        self._plot(plt.scatter, x, y=y, xlabel=xlabel, ylabel=ylabel, title=title, **kwargs) 
