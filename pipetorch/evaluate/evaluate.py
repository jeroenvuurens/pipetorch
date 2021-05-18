import numpy as np
import pandas as pd
import copy
import itertools
import matplotlib.pyplot as plt
from operator import itemgetter
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from .evaluateresults import EvaluatorResults

class Evaluator:
    def __init__(self, df, *metrics):
        self.df = df
        self.metrics = metrics
        self.results = EvaluatorResults.from_evaluator(self)
    
    def append(self, evaluator):
        r = copy.copy(self)
        r.results = r.results.append(evaluator.results, sort=True, ignore_index=True)
        return r
    
    def __repr__(self):
        return repr(self.results)
    
    def _1d(self, y):
        if y is None:
            return y
        try:
            y = y.cpu().numpy()
        except: pass
        return y.reshape(-1) if len(y.shape) > 1 else y
    
    @property
    def df(self):
        return self._df
    
    @df.setter
    def df(self, df):
        self._df = df
    
    @property
    def train_X(self):
        return self.df.train_X

    @property
    def train_y(self):
        return self.df.train_y

    @property
    def valid_X(self):
        return self.df.valid_X

    @property
    def valid_y(self):
        return self.df.valid_y

    @property
    def test_X(self):
        return self.df.test_X

    @property
    def test_y(self):
        return self.df.test_y
    
    @property
    def train(self):
        return self.results.train
    
    @property
    def valid(self):
        return self.results.valid
    
    @property
    def test(self):
        return self.results.test
    
    def __getitem__(self, key):
        return self.results[key]
    
    def compute_metrics(self, true_y, pred_y):
        pred_y = self._1d(pred_y)
        return {m.__name__: m(true_y, pred_y) for m in self.metrics}

    def _dict_to_df(self, *dicts):
        return pd.concat([pd.DataFrame(d, index=[0]) for d in dicts], axis=1)

    def run(self, train, predict, model=None, df=None, **annot):
        if df is None:
            df = self.df
        train(df.train_X, df.train_y)
        annot['_model'] = model
        annot['_predict'] = None if model is None else predict
        annot['_train'] = None if model is None else train
        self.score_train(predict, df=df, **annot)
        self.score_valid(predict, df=df, **annot)

    def run_sklearn(self, model, df=None, **annot):
        self.run(model.fit, model.predict, model=model, df=df, **annot)

    def _inverse_transform_y(self, df, y):
        if callable(getattr(df, "inverse_transform_y", None)):
            return df.inverse_transform_y( y )
        return y
        
    def _run(self, predict, X, y, df=None, **annot):
        y_pred = predict(X)
        self._store(y, y_pred, df=df, **annot)

    def _store(self, y, y_pred, df=None, **annot):
        if df is None:
            df = self.df
        y = self._inverse_transform_y( df, y )
        y_pred = self._inverse_transform_y( df, y_pred )
        metrics = self.compute_metrics(y, y_pred)
        self.results = self.results._add(self._dict_to_df(metrics, annot))

    def score_train(self, predict, df=None, **annot):
        if df is None:
            df = self.df
        self._run(predict, df.train_X, df.train_y, phase='train', df=df, **annot)

    def score_valid(self, predict, df=None, **annot):
        if df is None:
            df = self.df
        if len(df.valid_X) > 0:
            self._run(predict, df.valid_X, df.valid_y, phase='valid', df=df, **annot)
        
    def score_test(self, predict, df=None, **annot):
        if df is None:
            df = self.df
        if len(df.test_X) > 0:
            self._run(predict, df.test_X, df.test_y, phase='test', df=df, **annot)
            
    def _order(self, X):
        return X[:, 0].argsort(axis=0)

    def _column_xy(self, x=None, y=None):
        if x is None:
            x = self.df._columnx[0]
        if y is None:
            y = self.df._columny[0]
        return x, y
            
    def _column_x2y(self, x1=None, x2=None, y=None):
        if x1 is None:
            x1 = self.df._columnx[0]
        if x2 is None:
            x2 = self.df._columnx[1]
        if y is None:
            y = self.df._columny[0]
        return x1, x2, y
    
    def _graph_coords_callable(self, df, y):
        if callable(y):
            return self.df.inverse_transform_y( y(df.X) ).to_numpy()
        elif type(y) == str:
            return np.squeeze(df[[y]].to_numpy())
        return y
    
    def _graph_coords(self, df, *x):
        return  [ self._graph_coords_callable(df, c) for c in x ]         

    def _figure(self, x=None, y=None, xlabel = None, ylabel = None, title = None, interpolate=0, phase='train', df=None ):
        if df is None:
            df = self.df.train
        if interpolate > 0:
            assert (y is None) or (type(y)==str) or callable(y), 'You cannot interpolate with given results'
            df = df.interpolate(interpolate)
        x, y = self._column_xy(x, y)
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
        if title is not None:
            plt.title(title)
        if type(ylabel) == str:
            plt.ylabel(ylabel) 
        if type(xlabel) == str:
            plt.xlabel(xlabel)
        graph_x, graph_y = self._graph_coords(df, x, y)
        return df.X, graph_x, graph_y

    def _figure2d(self, x1=None, x2=None, y=None, xlabel = None, ylabel = None, title = None, df=None ):
        if df is None:
            df = self.df.train
        x1, x2, y = self._column_x2y(x1, x2, y)
        if not xlabel:
            if type(x1) == str:
                xlabel=x1
            else:
                xlabel=self.df._columnx[0]
        if not ylabel:
            if type(x2) == str:
                ylabel=x2
            else:
                ylabel=self.df._columnx[1]
        if title is not None:
            plt.title(title)
        if type(ylabel) == str:
            plt.ylabel(ylabel) 
        if type(xlabel) == str:
            plt.xlabel(xlabel)
        graph_x1, graph_x2, graph_y = self._graph_coords(df, x1, x2, y)
        return df.X, graph_x1, graph_x2, graph_y

    def _add_noise(self, x1, x2, noise):
        if noise > 0:
            x1_sd = np.std(x1)
            x2_sd = np.std(x2)
            x1 = x1 + np.random.normal(0, noise * x1_sd, x1.shape)
            x2 = x2 + np.random.normal(0, noise * x2_sd, x2.shape)
        return x1, x2
    
    def scatter2d_class(self, x1=None, x2=None, y=None, xlabel=None, ylabel=None, title=None, loc='upper right', noise=0, df=None, **kwargs):
        X, xd1, xd2, y = self._figure2d(x1=x1, x2=x2, y=y, xlabel=xlabel, ylabel=ylabel, title=title, df=df)
        for c in sorted(np.unique(y)):
            indices = (c == y).flatten()
            xx1 = xd1[indices].flatten()
            xx2 = xd2[indices].flatten()
            xx1, xx2 = self._add_noise(xx1, xx2, noise)
            plt.scatter(xx1, xx2, label=int(c), **kwargs)
        plt.gca().legend(loc=loc)

    def scatter2d_color(self, x1=None, x2=None, c=None, xlabel=None, ylabel=None, title=None, loc='upper right', noise=0, df=None, cmap=plt.get_cmap("jet"), s=1, **kwargs):
        X, x1, x2, c = self._figure2d(x1=x1, x2=x2, y=c, xlabel=xlabel, ylabel=ylabel, title=title, df=df)
        x1, x2 = self._add_noise(x1.flatten(), x2.flatten(), noise)
        plt.scatter(x1, x2, c=c, cmap=cmap, s=s, **kwargs)
        plt.colorbar()
        
    def scatter2d_size(self, x1=None, x2=None, s=None, xlabel=None, ylabel=None, title=None, loc='upper right', noise=0, df=None, **kwargs):
        X, x1, x2, s = self._figure2d(x1=x1, x2=x2, y=s, xlabel=xlabel, ylabel=ylabel, title=title, df=df)
        x1, x2 = self._add_noise(x1.flatten(), x2.flatten(), noise)
        plt.scatter(x1, x2, s=s, **kwargs)
        
    def _boundaries(self, predict):
        ax = plt.gca()
        x_min, x_max = ax.get_xlim()
        y_min, y_max = ax.get_ylim()
        stepx = (x_max - x_min) / 600
        stepy = (y_max - y_min) / 600
        xx, yy = np.meshgrid(np.arange(x_min, x_max, stepx),
                             np.arange(y_min, y_max, stepy))
        X = np.array(np.vstack([xx.ravel(), yy.ravel()])).T
        s = self.df.from_numpy(X)
        try:
            return ax, xx, yy, predict(s.X).reshape(xx.shape)       
        except:
            try:
                import torch
                with torch.set_grad_enabled(False):
                    return ax, xx, yy, predict(s.X_tensor).numpy().reshape(xx.shape)
            except:
                raise ValueError('predict mus be a function that works on Numpy arrays or PyTorch tensors')
    
    def plot_boundary(self, predict):
        ax, xx, yy, boundary = self._boundaries(predict)
        ax.contour(xx, yy, boundary, levels=[0.5])  
        
    def plot_contour(self, predict):
        ax, xx, yy, boundary = self._boundaries(predict)
        ax.contour(xx, yy, boundary)
        
    def _coef(self, coefs, subset, model):
        try:
            model.fit(self.df.train_X[subset], self.df.train_y[subset])
            coefs.append( (model.intercept_, self.get_coef(model.coef_ ) ) )
        except:pass
    
    def _loss_minmax(self, model, loss):
        model.fit(self.df.train_X, self.df.train_y)
        min0 = model.intercept_
        min1 = self.get_coef(model.coef_ )
        m = np.argsort(self.df.train_X[:, 0])
        n = len(m)//2
        coefs = []
        self._coef(coefs, m[:n], model)
        self._coef(coefs, m[n:], model)
        m = np.argsort(self.df.train_y)
        self._coef(coefs, m[:n], model)
        self._coef(coefs, m[n:], model)
        model.fit(self.df.train_X, self.df.train_y)
        max0 = max([ a for a, _ in coefs ])
        max1 = max([ b for _, b in coefs ])
        l = self._loss_coef_intercept(model, loss, min0, min1)
        loss0 = self._loss_coef_intercept(model, loss, max0, min1)
        loss1 = self._loss_coef_intercept(model, loss, min0, max1)
        if (loss0 - l) < (loss1 - l)/ 2:
            while (loss0 - l) < (loss1 - l)/ 2:
                max0 *= 2
                loss0 = self._loss_coef_intercept(model, loss, max0, min1)
        elif (loss1 - l) < (loss0 - l) / 2:
            while (loss1 - l) < (loss0 - l) / 2:
                max1 *= 2
                loss1 = self._loss_coef_intercept(model, loss, min0, max1)
        if max0 > min0:
            min0 = min0 - (max0 - min0)
        else:
            min0, max0 = max0, min0 + (min0 - max0)

        if max1 > min1:
            min1 = min1 - (max1 - min1)
        else:
            min1, max1 = max1, min1 + (min1 - max1)
        return min0, max0, min1, max1

    def _loss_coef_intercept(self, model, loss, intercept, coef):
        self.set_coef( model.coef_, coef )
        model.intercept_ = intercept
        pred_y = model.predict(self.df.train_X)
        return loss(self.df.train_y, pred_y)

    def get_coef(self, coef):
        if len(coef.shape) > 1:
            return self.get_coef(coef[0])
        return coef[0]
    
    def set_coef(self, coef, value):
        if len(coef.shape) > 1:
            self.set_coef(coef[0], value)
        coef[0] = value
    
    def loss_surface(self, model, loss, linewidth=1, antialiased=False, cmap=cm.coolwarm, intersects=50, **kwargs):
        model = copy.copy(model)
        min0, max0, min1, max1 = self._loss_minmax(model, loss)
        step0 = np.abs(max0 - min1) / intersects
        step1 = np.abs(max1 - min1) / intersects
        xx, yy = np.meshgrid(np.arange(min0, max0, step0),
                             np.arange(min1, max1, step1))
        X = np.array(np.vstack([xx.ravel(), yy.ravel()])).T
        l = np.array([ self._loss_coef_intercept(model, loss, intercept, coef) for intercept, coef in X ])
        l = l.reshape(xx.shape)
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.plot_surface(xx, yy, l, cmap=cmap, linewidth=linewidth, antialiased=antialiased)
        plt.xlabel(r'$\theta_0$')
        plt.ylabel(r'$\theta_1$')
        
    def _plot(self, pltfunction, x=None, y=None, xlabel = None, ylabel = None, title=None, marker=None, interpolate=0, df=None, **kwargs):
        X, gx, gy = self._figure(x=x, y=y, xlabel=xlabel, ylabel=ylabel, title=title, interpolate=interpolate, df=df)
        pltfunction(gx, gy, marker=marker, **kwargs)
        if 'label' in kwargs:
            plt.legend()
    
    def line(self, x=None, y=None, xlabel = None, ylabel = None, title=None, interpolate=0, df=None, **kwargs):
        self._plot(plt.plot, x=x, y=y, xlabel=xlabel, ylabel=ylabel, title=title, interpolate=interpolate, df=df, **kwargs)

    def scatter(self, x=None, y=None, xlabel = None, ylabel = None, title=None, interpolate=0, df=None, **kwargs):
        self._plot(plt.scatter, x=x, y=y, xlabel=xlabel, ylabel=ylabel, title=title, interpolate=interpolate, df=df, **kwargs)
        
  #  def line_metric(self, x, y=None, xlabel = None, ylabel = None, title=None, **kwargs):
  #      
  #      self.results.train.line(x, y=y, xlabel=xlabel, ylabel=ylabel, title=title, **kwargs)
  #      self.results.valid.line(x, y=y, xlabel=xlabel, ylabel=ylabel, title=title, **kwargs)
  #      plt.legend()
       
    def _groups(self, series='phase', select=None):
        if select is None:
            s = self.results
        elif type(select) is pd.core.series.Series:
            s = self.results[select]
        elif type(select) is EvaluatorResults:
            s = select
        else:
            raise ValueError('Unknown type passed for select')
        for g, d in s.groupby(by=series):
            yield g, self.results._copy_meta(d)
    
    def scatter_metric(self, x, series='phase', select=None, y=None, xlabel = None, ylabel = None, title=None, **kwargs):
        for g, d in self._groups(series=series, select=select):
            d.scatter(x, y=y, xlabel=xlabel, ylabel=ylabel, title=title, label=label_prefix + str(g), **kwargs)
        plt.legend()
    
    def line_metric(self, x, series='phase', select=None, y=None, xlabel = None, ylabel = None, title=None, label_prefix='', **kwargs):
        for g, d in self._groups(series=series, select=select):
            d.line(x, y=y, xlabel=xlabel, ylabel=ylabel, title=title, label=label_prefix + str(g), **kwargs)
        plt.legend()