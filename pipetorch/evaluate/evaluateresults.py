import numpy as np
import pandas as pd
import copy
import itertools
import matplotlib.pyplot as plt

class EvaluatorResults(pd.DataFrame):
    _metadata = ['evaluator', '_phase', 'df']
    
    @classmethod
    def from_evaluator(cls, 
                       evaluator):
        r = cls()
        r.evaluator = evaluator
        r.df = evaluator.df
        return r
    
    @property
    def _constructor(self):
        return EvaluatorResults

    def _copy_meta(self, 
                   df):
        df = self._constructor(df)
        for m in self._metadata:
            setattr(df, m, getattr(self, m))
        return df
    
    def clone(self):
        return copy.copy(self)
    
    def __finalize__(self, 
                     other, 
                     method=None, 
                     **kwargs):
        for name in self._metadata:
            object.__setattr__(self, name, getattr(other, name, None))
        return self
    
    def __repr__(self):
        r = self.drop( columns = [ c for c in self.columns if c[0] == '_' ] )
        return super(EvaluatorResults, r).__repr__()
    
    def _add(self, 
             row):
        for c in row.columns:
            if c not in self.columns:
                self.insert(len(self.columns), c, np.NaN)
        r = pd.concat([self, row], sort=True, ignore_index=True)
        #r = self.append(row, sort=True, ignore_index=True)
        r.evaluator = self.evaluator
        return r

    @property
    def train(self):
        r = self.loc[self.phase == 'train'].copy()
        r._phase = 'train'
        return r
    
    @property
    def valid(self):
        r = self.loc[self.phase == 'valid'].copy()
        r._phase = 'valid'
        return r
    
    @property
    def test(self):
        r = self.loc[self.phase == 'test'].copy()
        r.phase = 'test'
        return r

    def _plot_predict(self, 
                      pltfunction, 
                      label, 
                      x=None, 
                      xlabel = None, 
                      ylabel = None, 
                      title=None, 
                      interpolate=0, 
                      df=None, 
                      **kwargs):
        marker = itertools.cycle((',', '+', '.', 'o', '*')) 
        for _, row in self.iterrows():
            y = row['_predict']
            try:
                l = label(row)
            except:
                l = row[label]
            self.evaluator._plot(pltfunction, x=x, y=y, xlabel=xlabel, ylabel=ylabel, title=title, marker=next(marker), interpolate=interpolate, df=df, label=l, **kwargs)
        plt.legend()
   
    def _select(self, 
                select=None):
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
    
    def _plot(self, 
              pltfunction, 
              x, y=None, 
              xlabel = None, 
              ylabel = None, 
              title=None, 
              loc='best', 
              fig=None, 
              legendargs={}, 
              **kwargs):
        f = _figure(self, x=x, y=y, xlabel=xlabel, ylabel=ylabel, title=title, fig=fig)
        pltfunction(f.graph_x, f.graph_y, **kwargs)      
        if 'label' in kwargs:
            self.evaluator._legend(f.fig, loc, **legendargs)

    def line(self, 
             x, 
             y=None, 
             xlabel = None, 
             ylabel = None, 
             title=None, 
             fig=None, 
             legendargs={}, 
             **kwargs):
        fig = fig or plt
        self._plot(fig.plot, x, y=y, xlabel=xlabel, ylabel=ylabel, title=title, 
                   fig=fig, legendargs=legendargs, **kwargs)
    
    def scatter(self, 
                x, 
                y=None, 
                xlabel = None, 
                ylabel = None, 
                title=None, 
                fig=None, 
                legendargs={}, 
                **kwargs):
        fig = fig or plt
        self._plot(fig.scatter, x, y=y, xlabel=xlabel, ylabel=ylabel, 
                   title=title, fig=fig, legendargs=legendargs, **kwargs)
        
    def line_metric(self, 
                    x, 
                    series='phase', 
                    y=None, 
                    xlabel = None, 
                    ylabel = None, 
                    title=None, 
                    label_prefix='', 
                    fig=None,
                    legendargs={},
                    **kwargs):
        self.evaluator.line_metric(x, series=series, select=self, y=y, xlabel=xlabel, ylabel=ylabel, 
                                   title=title, label_prefix=label_prefix, fig=fig, legendargs=legendargs, **kwargs)

    def scatter_metric(self, 
                       x, 
                       series='phase', 
                       y=None, 
                       xlabel = None, 
                       ylabel = None, 
                       title=None, 
                       label_prefix='', 
                       fig=None,
                       legendargs={},
                       **kwargs):
        self.evaluator.scatter_metric(x, series=series, select=self, y=y, xlabel=xlabel, ylabel=ylabel, 
                                      title=title, label_prefix=label_prefix, fig=fig, legendargs=legendargs, **kwargs)

class _figure:
    def __init__(self, 
                 results, 
                 x=None, 
                 y=None, 
                 xlabel = None, 
                 ylabel = None, 
                 title = None, 
                 fig=None ):
        fig = fig or plt
        self.evaluator = results.evaluator
        self.results = results
        self.x = x
        self.y = y
        self.xlabel = xlabel or self.x
        self.ylabel = ylabel or self.y
        self.fig = fig
        if title is not None:
            try:
                fig.title(title)
            except:
                fig.suptitle(title)
        try:
            fig.ylabel(self.ylabel)
        except:
            fig.set_ylabel(self.ylabel)
        try:
            fig.xlabel(self.xlabel)
        except:
            fig.set_xlabel(self.xlabel)
        self.results = self.results.sort_values(by=self.x)
        gx = [ row[self.x] for _, row in self.results.iterrows() ]
        gy = [ row[self.y] for _, row in self.results.iterrows() ]

    @property
    def x(self):
        return self._x
    
    @x.setter
    def x(self, 
          value):
        self._x = value or self.df._columnx[0]
        
    @property
    def y(self):
        return self._y
    
    @y.setter
    def y(self, 
          value):
        self._y = value or self.evaluator.metrics[0].__name__
        
    @property
    def xlabel(self):
        return self._xlabel
    
    @xlabel.setter
    def xlabel(self, 
               value):
        self._xlabel = value or self.x

    @property
    def ylabel(self):
        return self._ylabel
    
    @ylabel.setter
    def ylabel(self, 
               value):
        self._ylabel = value or self.y

    @property
    def graph_x(self):
        return [ row[self.x] for _, row in self.results.iterrows() ]

    @property
    def graph_y(self):
        return [ row[self.y] for _, row in self.results.iterrows() ]
