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

    def scatter2d_class(self, x1=None, x2=None, y=None, xlabel=None, ylabel=None, title=None, loc='upper right', noise=0, df=None, **kwargs):
        f = _figure2d(self, x1=x1, x2=x2, y=y, xlabel=xlabel, ylabel=ylabel, title=title, df=df, noise=noise)
        for c in sorted(np.unique(f.graph_y)):
            indices = (c == f.graph_y)
            plt.scatter(f.graph_x1[indices], f.graph_x2[indices], label=int(c), **kwargs)
        plt.gca().legend(loc=loc)

    def scatter2d_color(self, x1=None, x2=None, c=None, xlabel=None, ylabel=None, title=None, noise=0, df=None, cmap=plt.get_cmap("jet"), s=1, **kwargs):
        f = _figure2d(self, x1=x1, x2=x2, y=c, xlabel=xlabel, ylabel=ylabel, title=title, df=df, noise=noise)
        plt.scatter(f.graph_x1, f.graph_x2, c=f.graph_y, cmap=cmap, s=s, **kwargs)
        plt.colorbar()
        
    def scatter2d_size(self, x1=None, x2=None, s=None, xlabel=None, ylabel=None, title=None, noise=0, df=None, **kwargs):
        f = _figure2d(self, x1=x1, x2=x2, y=s, xlabel=xlabel, ylabel=ylabel, title=title, df=df, noise=noise)
        plt.scatter(f.graph_x1, f.graph_x2, s=f.graph_y, **kwargs)
        
    def _boundaries(self, predict):
        ax = plt.gca()
        x_min, x_max = ax.get_xlim()
        y_min, y_max = ax.get_ylim()
        stepx = (x_max - x_min) / 600
        stepy = (y_max - y_min) / 600
        xx, yy = np.meshgrid(np.arange(x_min, x_max, stepx),
                             np.arange(y_min, y_max, stepy))
        X = np.array(np.vstack([xx.ravel(), yy.ravel()])).T
        print(X.shape, self.df._columnx)
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
        
    def _plot(self, pltfunction, x=None, y=None, xlabel = None, ylabel = None, sort=False, title=None, marker=None, interpolate=0, df=None, loc='upper right', **kwargs):
        f = _figure(self, x=x, y=y, xlabel=xlabel, ylabel=ylabel, title=title, sort=sort, interpolate=interpolate, df=df)
        pltfunction(f.graph_x, f.graph_y, marker=marker, **kwargs)
        if 'label' in kwargs:
            plt.legend(loc=loc)
    
    def line(self, x=None, y=None, xlabel = None, ylabel = None, title=None, interpolate=0, df=None, loc='upper right', **kwargs):
        self._plot(plt.plot, x=x, y=y, xlabel=xlabel, ylabel=ylabel, title=title, interpolate=interpolate, sort=True, df=df, loc=loc, **kwargs)

    def scatter(self, x=None, y=None, xlabel = None, ylabel = None, title=None, interpolate=0, df=None, loc='upper right', **kwargs):
        self._plot(plt.scatter, x=x, y=y, xlabel=xlabel, ylabel=ylabel, title=title, interpolate=interpolate, df=df, loc=loc, **kwargs)
       
    def _select(self, select):
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

    def _unique(self, selection, series='phase'):
        return len(selection[series].unique())
    
    def _groups(self, selection, series='phase'):
        for g, d in selection.groupby(by=series):
            yield g, self.results._copy_meta(d)
    
    def scatter_metric(self, x, series='phase', select=None, y=None, xlabel = None, ylabel = None, title=None, label_prefix='', label=None, **kwargs):
        selection = self._select(select)
        unique_groups = self._unique(selection, series)
        for g, d in self._groups(selection, series=series):
            g = label or (label_prefix + str(g) if unique_groups > 1 else ylabel)
            d.scatter(x, y=y, xlabel=xlabel, ylabel=ylabel, title=title, label=g, **kwargs)
    
    def line_metric(self, x, series='phase', select=None, y=None, xlabel = None, ylabel = None, title=None, label_prefix='', label=None, **kwargs):
        selection = self._select(select)
        unique_groups = self._unique(selection, series)
        for g, d in self._groups(selection, series=series):
            g = label or (label_prefix + str(g) if unique_groups > 1 else ylabel)
            d.line(x, y=y, xlabel=xlabel, ylabel=ylabel, title=title, label=g, **kwargs)
        
class _figures:
    def _graph_coords_callable(self, df, f):
        if callable(f):
            return self.evalator.df.inverse_transform_y( f(df.X) ).to_numpy()
        elif type(f) == str:
            return np.squeeze(df[[f]].to_numpy())
        return f
    
    def _graph_coords(self, df, *fields):
        return  [ self._graph_coords_callable(df, f) for f in fields ]         
        
class _figure(_figures):
    def __init__(self, evaluator, x=None, y=None, xlabel = None, ylabel = None, title = None, sort=False, interpolate=0, phase='train', df=None ):
        self.evaluator = evaluator
        self.df = df if df is not None else evaluator.df.train
        self.x = x
        self.y = y
        self.xlabel = xlabel
        self.ylabel = ylabel
        if title is not None:
            plt.title(title)
        plt.ylabel(self.ylabel) 
        plt.xlabel(self.xlabel)
        if interpolate > 0:
            assert (y is None) or (type(y)==str) or callable(y), 'You cannot interpolate with given results'
            self.df = self.df.interpolate(interpolate)
        elif sort:
            self.df = self.df.sort_values(by=x)

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
        self._y = value or self.df._columny[0]
        
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
        return self._graph_coords_callable(self.df, self.x)

    @property
    def graph_y(self):
        return self._graph_coords_callable(self.df, self.y)

    @property
    def X(self):
        return self.df.X

class _figure2d(_figures):
    def __init__(self, evaluator, x1=None, x2=None, y=None, xlabel = None, ylabel = None, title = None, df=None, noise=0 ):
        self.evaluator = evaluator
        self.df = df if df is not None else evaluator.df.train
        self.noise = noise
        self.x1 = x1
        self.x2 = x2
        self.y = y
        self.xlabel = xlabel
        self.ylabel = ylabel
        if title is not None:
            plt.title(title)
        plt.ylabel(ylabel) 
        plt.xlabel(xlabel)

    @property
    def x1(self):
        return self._x1
    
    @x1.setter
    def x1(self, value):
        self._x1 = value or self.df._columnx[0]

    @property
    def x2(self):
        return self._x2
    
    @x2.setter
    def x2(self, value):
        self._x2 = value or self.df._columnx[1]
        
    @property
    def y(self):
        return self._y
    
    @y.setter
    def y(self, value):
        self._y = value or self.df._columny[0]
        
    @property
    def xlabel(self):
        return self._xlabel
    
    @xlabel.setter
    def xlabel(self, value):
        self._xlabel = value or self.x1

    @property
    def ylabel(self):
        return self._ylabel
    
    @ylabel.setter
    def ylabel(self, value):
        self._ylabel = value or self.x2

    @property
    def graph_x1_noiseless(self):
        return self._graph_coords_callable(self.df, self.x1)

    @property
    def graph_x2_noiseless(self):
        return self._graph_coords_callable(self.df, self.x2)

    @property
    def graph_y(self):
        return self._graph_coords_callable(self.df, self.y)

    @property
    def graph_x1(self):
        try:
            return self._graph_x1
        except:
            self._graph_x1 = self.graph_x1_noiseless
            if self.noise > 0:
                # should we flatten??
                sd = np.std(self._graph_x1)
                self._graph_x1 = self._graph_x1 + np.random.normal(0, self.noise * sd, self._graph_x1.shape)
            return self._graph_x1

    @property
    def graph_x2(self):
        try:
            return self._graph_x2
        except:
            self._graph_x2 = self.graph_x2_noiseless
            if self.noise > 0:
                sd = np.std(self._graph_x2)
                self._graph_x2 = self._graph_x2 + np.random.normal(0, self.noise * sd, self._graph_x2.shape)
            return self._graph_x2
    
    @property
    def X(self):
        return self.df.X