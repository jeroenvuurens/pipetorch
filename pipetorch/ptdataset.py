import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
import matplotlib.pyplot as plt
import copy
import path
from pathlib import Path
import os

class PTDataSet(pd.DataFrame):
    _metadata = ['_df', '_pt_scalerx', '_pt_scalery', '_pt_columny', '_pt_transposey', '_pt_bias', '_pt_polynomials', '_pt_evaluator']

#     _internal_names = pd.DataFrame._internal_names + ['__pt_x_numpy', '__pt_x_polynomials', '__pt_x_scaled', '__pt_x_biased', '__pt_scale_columns', '_pt__y_numpy', '_pt__y_scaled', '_pt__y_transposed']
#     _internal_names_set = set(_internal_names)

    @property
    def _constructor(self):
        return PTDataSet
       
    @classmethod
    def from_ptdataframe(cls, data, df):
        r = cls(data)
        r._df = df
        return r

    @property
    def _scalerx(self):
        return self._pt_scalerx
        
    @property
    def _scalery(self):
        return self._pt_scalery

    @property
    def _columny(self):
        return [ self.columns[-1] ] if self._pt_columny is None else self._pt_columny
        
    @property
    def _transposey(self):
        return True if self._pt_transposey is None else self._pt_transposey
            
    @property
    def _columnx(self):
        return [ c for c in self.columns if c not in self._columny ]
   
    @property
    def _polynomials(self):
        return self._pt_polynomials
       
    @property
    def _bias(self):
        return self._pt_bias
    
    def _transform(self, scalers, array):
        out = []
        for i, scaler in enumerate(scalers):
            if scaler is not None:
                out.append(scaler.transform(array[:, i:i+1]))
            else:
                out.append(array[:, i:i+1])
        return np.concatenate(out, axis=1)

    def interpolate(self, factor=2, sortcolumn=None):
        if not sortcolumn:
            sortcolumn = self.columns[0]
        df = self.sort_values(by=sortcolumn)
        for i in range(factor):
            i = df.rolling(2).sum()[1:] / 2.0
            df = pd.concat([df, i], axis=0)
            df = df.sort_values(by=sortcolumn)
        return self._df._ptdataset(df)
    
    @property
    def _x_numpy(self):
        return self[self._columnx].to_numpy()
    
    @property
    def _x_polynomials(self):
        try:
            return self._polynomials.fit_transform(self._x_numpy)
        except:
            return self._x_numpy

    @property
    def _x_scaled(self):
        if len(self > 0):
            return self._transform(self._scalerx(), self._x_polynomials)
        return self._x_polynomials
            
    @property
    def _x_biased(self):
        a = self._x_scaled
        if self._bias:
            return np.concatenate([np.ones((len(a),1)), a], axis=1)
        return a

    @property
    def X(self):
        return self._x_biased

    @property
    def _y_numpy(self):
        return self[self._columny].to_numpy()
    
    @property
    def _y_scaled(self):
        if len(self) > 0:
            return self._transform(self._scalery(), self._y_numpy)
        return self._y_numpy
            
    @property
    def _y_transposed(self):
        return self._y_scaled.squeeze() if self._transposey else self._y_scaled

    @property
    def y(self):
        return self._y_transposed
    
    def line(self, x=None, y=None, xlabel = None, ylabel = None, title = None, **kwargs ):
        self._df.evaluate().line(x=x, y=y, xlabel=xlabel, ylabel=ylabel, title=title, df=self, **kwargs)
    
    def scatter(self, x=None, y=None, xlabel = None, ylabel = None, title = None, **kwargs ):
        self._df.evaluate().scatter(x=x, y=y, xlabel=xlabel, ylabel=ylabel, title=title, df=self, **kwargs)
    
    def scatter2d_class(self, x1=None, x2=None, y=None, xlabel=None, ylabel=None, title=None, markersize=2, loc='upper right', noise=0, **kwargs):
        self._df.evaluate().scatter2d_class(x1=x1, x2=x2, y=y, xlabel=xlabel, ylabel=ylabel, title=title, markersize=markersize, loc=loc, noise=noise, df=self, **kwargs)

    def scatter2d_color(self, x1=None, x2=None, c=None, xlabel=None, ylabel=None, title=None, markersize=2, loc='upper right', noise=0, **kwargs):
        self._df.evaluate().scatter2d_color(x1=x1, x2=x2, c=c, xlabel=xlabel, ylabel=ylabel, title=title, markersize=markersize, loc=loc, noise=noise, df=self, **kwargs)

    def scatter2d_size(self, x1=None, x2=None, s=None, xlabel=None, ylabel=None, title=None, markersize=2, loc='upper right', noise=0, **kwargs):
        self._df.evaluate().scatter2d_size(x1=x1, x2=x2, s=s, xlabel=xlabel, ylabel=ylabel, title=title, markersize=markersize, loc=loc, noise=noise, df=self, **kwargs)

    def plot_boundary(self, predict):
        self._df.evaluate().plot_boundary(predict)
        
    def plot_contour(self, predict):
        self._df.evaluate().plot_contour(predict)
