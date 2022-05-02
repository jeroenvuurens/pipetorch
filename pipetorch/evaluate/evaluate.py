import numpy as np
import pandas as pd
import copy
import itertools
import matplotlib.pyplot as plt
from operator import itemgetter
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from .evaluateresults import EvaluatorResults
from collections.abc import Iterable

class Evaluator:
    def __init__(self, df, *metrics):
        self.df = df
        self.metrics = metrics
        self.reset()   # create a fresh set of results
    
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
            y = y.cpu()
        except: pass
        try:
            y = y.numpy()
        except: pass
        try:
            y = y.to_numpy()
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
        if len(self.metrics) > 0:
            return {m.__name__: m(true_y, pred_y) for m in self.metrics}
        return dict()

    def _dict_to_df(self, *dicts):
        r = []
        for d in dicts:
            d = { key:([value] if isinstance(value, Iterable) else value) for key,value in d.items() }
            r.append( pd.DataFrame(d, index=[0]))
        return pd.concat(r, axis=1)
    
    def reset(self):
        self.results = EvaluatorResults.from_evaluator(self)
    
    def optimum(self, *targets, direction=None, directions=None, validation=None, test=None, select=None):
        """
        Finds the optimal value in a training trial, i.e. where the given `optimize` value over the validation
        set is optimal. The optimize metric must be among the metrics that are cached.
        
        When there are multiple target metrics, ties on the first metric are resolved by the second, etc. 
        
        Arguments:
            *targets: str
                one or more target metrics that are used to find the optimum
                if empty, all recorded metrics are used in the order they were registered (usually loss first)
            direction: str ('minimize')
                'minimize' or 'maximize' to return the resp. lowest or highest score on the validation set
            directions: [ str ] ('minimize')
                for multi-target training
            validation, test: 
                for internal use, allows recursive calls to resolve finding a multi-target optimum
        
        Returns:
            { metric:value }
            A dictionary of values obtained over the cached metrics on the validation set
        """
        
        if validation is None:
            validation = self._select(select).valid
        if test is None:
            test = self._select(select).test if 'test' in self.results.phase.unique() else validation
        if len(targets) == 0:
            return self.optimum(*self.metrics, direction=direction, directions=directions, validation=validation, test=test)
        if directions is None and direction is None:
            return self.optimum(*targets, directions=[ 'minimize' if t == 'loss' else 'maximize' for t in targets ], validation=validation, test=test)
        if direction is not None:
            assert directions is None, 'You can only use direction or directions'
            return self.optimum(*targets, directions=[ direction ], validation=validation, test=test)
        for t in targets:
            assert t in validation.metric.unique(), f'target {t} must be a cached metric'
        m = validation[validation.metric == targets[0]]
        if directions[0] == 'minimize':
            optimum = m.value.min()
        elif directions[0] == 'maximize':
            optimum = m.value.max()
        else:
            assert False, 'direction must be minimize or maximize'
        epochs = m.epoch[m.value == optimum]
        test = test[test.epoch.isin(epochs)]
        if len(epochs) > 1 and len(targets) > 1:
            validation = validation[validation.epoch.isin(epochs)]
            return self.optimum( *target[1:], directions=directions[1:], validation=validation, test=test)
        elif len(epochs) > 1:
            test = test.loc[test.epoch == epochs[0]]
        return {t:test.value[test.metric == t].item() for t in targets }
    
    def train_valid_sklearn(self, df):
            model.fit(df.train_X, df.train_y)
            valid_pred = df.valid._predict(predict)
            valid_y = df._inverse_transform_y( df.valid_y )
            valid_pred = df._inverse_transform_y( valid_pred )
            metrics = self.compute_metrics(valid_y, valid_pred)
            for k, v in metrics.items():
                results_valid[k] += v * len(df.valid_X) / len(df)
            train_pred = dfk.train._predict(predict)
            y = dfk._inverse_transform_y( y )
            y_pred = dfk._inverse_transform_y( y_pred )
            metrics = self.compute_metrics(y, y_pred)
            for k, v in metrics.items():
                results_train[k] += v  * len(dfk.train_X) / len(df)
    
    def run(self, train, predict, model=None, df=None, n_splits=1, stratify=None, **annot):
        if df is None:
            df = self.df
        if n_splits == 1:
            train(df.train_X, df.train_y)
            self.score_train(predict, df=df, **annot)
            self.score_valid(predict, df=df, **annot)
        else:
            results_train = defaultdict(lambda:0)
            results_valid = defaultdict(lambda:0)
            for dfk in df.kfold(n_splits=n_splits, stratify=stratify):
                train(dfk.train_X, dfk.train_y)
                y_pred = dfk.valid._predict(predict)
                y = dfk._inverse_transform_y( y )
                y_pred = dfk._inverse_transform_y( y_pred )
                metrics = self.compute_metrics(y, y_pred)
                for k, v in metrics.items():
                    results_valid[k] += v * len(dfk.valid_X) / len(df)
                y_pred = dfk.train._predict(predict)
                y = dfk._inverse_transform_y( y )
                y_pred = dfk._inverse_transform_y( y_pred )
                metrics = self.compute_metrics(y, y_pred)
                for k, v in metrics.items():
                    results_train[k] += v  * len(dfk.train_X) / len(df)
            results_train['phase'] = 'train'
            results_valid['phase'] = 'valid'
            self.results = self.results._add(self._dict_to_df(train_results, annot))
            self.results = self.results._add(self._dict_to_df(valid_results, annot))
            
    def run_sklearn(self, model, df=None, **annot):
            
        self.run(model.fit, model.predict, model=model, df=df, **annot)

    def _inverse_transform_y(self, df, y):
        if callable(getattr(df, "inverse_transform_y", None)):
            return df.inverse_transform_y( y )
        return y
        
    def _store_predict(self, predict, df, **annot):
        y_pred = df._predict(predict)
        self._store(df.y, y_pred, df=df, **annot)        
        
    def _store_metrics(self, y, y_pred, df=None, **annot):
        if df is None:
            df = self.df
        y = self._inverse_transform_y( df, y )
        y_pred = self._inverse_transform_y( df, y_pred )
        metrics = self.compute_metrics(y, y_pred)
        for m, value in metrics.items():
            self._store_metric(m, value, **annot)
        return metrics

    def _store_metric(self, metric, value, **annot):
        self.results = self.results._add(self._dict_to_df({'metric':metric, 'value': value}, annot))
            
    def score_train(self, predict, df=None, **annot):
        if df is None:
            df = self.df
        self._store_predict(predict, df.train, phase='train', **annot)
            
    def score_valid(self, predict, df=None, **annot):
        if df is None:
            df = self.df
        if len(df.valid_X) > 0:
            self._store_predict(predict, df.valid, phase='valid', **annot)

    def score_test(self, predict, df=None, **annot):
        if df is None:
            df = self.df
        if len(df.test_X) > 0:
            self._store_predict(predict, df.test, phase='test', **annot)
                
    def _order(self, X):
        return X[:, 0].argsort(axis=0)

    def scatter2d_class(self, x1=None, x2=None, y=None, xlabel=None, ylabel=None, title=None, loc='best', noise=0, df=None, **kwargs):
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
        dataset = self.df.from_numpy(X)
        boundaries = dataset._predict(predict).to_numpy()
        boundaries.resize(xx.shape)
        return ax, xx, yy, boundaries
    
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
        min0 = min0 - (max0 - min0)
        min1 = min1 - (max1 - min1)
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
        step0 = np.abs(max0 - min0) / intersects
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
        plt.show()
        
    def _figure(self, x=None, y=None, xlabel = None, ylabel = None, sort=False, title=None, interpolate=0, df=None):
        return _figure(self, x=x, y=y, xlabel=xlabel, ylabel=ylabel, title=title, sort=sort, interpolate=interpolate, df=df)
    
    def _plot(self, pltfunction, x=None, y=None, xlabel = None, ylabel = None, sort=False, title=None, marker=None, interpolate=0, df=None, loc='best', **kwargs):
        #fig = plt.figure()
        f = _figure(self, x=x, y=y, xlabel=xlabel, ylabel=ylabel, title=title, sort=sort, interpolate=interpolate, df=df)
        pltfunction(f.graph_x, f.graph_y, marker=marker, **kwargs)
        if 'label' in kwargs:
            plt.gca().legend(loc=loc)
        #fig.show()
    
    def line(self, x=None, y=None, xlabel = None, ylabel = None, title=None, interpolate=0, df=None, loc='best', **kwargs):
        self._plot(plt.plot, x=x, y=y, xlabel=xlabel, ylabel=ylabel, title=title, interpolate=interpolate, sort=True, df=df, loc=loc, **kwargs)

    def scatter(self, x=None, y=None, xlabel = None, ylabel = None, title=None, interpolate=0, df=None, loc='best', **kwargs):
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
        elif type(select) == dict:
            s = self.results
            for key, value in select.items():
                s = s.loc[s[key] == value].copy()
        else:
            raise ValueError('Unknown type passed for select')
        return s

    def _groups(self, selection, series='phase'):
        for g, d in selection.groupby(by=series):
            yield g, self.results._copy_meta(d)
    
    def scatter_metric(self, x, y=None, series='phase', select=None, xlabel = None, ylabel = None, title=None, label_prefix='', label=None, **kwargs):
        y = y or self.metrics[0].__name__
        selection = self._select(select)
        selection = selection[selection.metric == y]
        ylabel = ylabel or y
        unique_groups = self._unique(selection, series)
        for g, d in self._groups(selection, series=series):
            g = label or (label_prefix + str(g) if unique_groups > 1 else ylabel)
            d.scatter(x, y='value', xlabel=xlabel, ylabel=y, title=title, label=g, **kwargs)
    
    def line_metric(self, x, y=None, series='phase', select=None, xlabel = None, ylabel = None, title=None, label_prefix='', label=None, **kwargs):
        y = y or self.metrics[0].__name__
        selection = self._select(select)
        selection = selection[selection.metric == y]
        ylabel = ylabel or y
        unique_groups = len([ a for a in self._groups(selection, series=series) ])
        for g, d in self._groups(selection, series=series):
            g = label or (label_prefix + str(g) if unique_groups > 1 else ylabel)
            d.line(x, y='value', xlabel=xlabel, ylabel=ylabel, title=title, label=g, **kwargs)
        
class _figures:
    def _graph_coords_callable(self, df, f):
        if callable(f):
            return self.evaluator.df.inverse_transform_y( f(df.X) ).to_numpy()
        elif type(f) == str:
            return np.squeeze(df[[f]].to_numpy())
        return f
    
    def _graph_coords(self, df, *fields):
        return  [ self._graph_coords_callable(df, f) for f in fields ]         
        
class _figure(_figures):
    def __init__(self, evaluator, x=None, y=None, xlabel = None, ylabel = None, title = None, sort=False, interpolate=0, phase='train', df=None ):
        self.evaluator = evaluator
        self.df = copy.copy(df if df is not None else evaluator.df.train)
        self.x = x
        self.xlabel = xlabel or self.x
        if interpolate > 0:
            assert (y is None) or (type(y)==str) or callable(y), 'You cannot interpolate with given results'
            self.df = self.df.interpolate_factor(interpolate)
        elif sort:
            self.df = self.df.sort_values(by=self.x)
        self.y = y
        self.ylabel = ylabel or self.y
        if title is not None:
            plt.title(title)
        plt.ylabel(self.ylabel) 
        plt.xlabel(self.xlabel)

    @property
    def x(self):
        return self._x
    
    @x.setter
    def x(self, value):
        if value is None:
            self._x = self.df._columnx[0]
        elif value is int:
            self._x = self.df._columnx[value]
        elif type(value) == str:
            self._x = value
        else:
            self._x = self.df._columnx[0]

    @property
    def y(self):
        return self._y
    
    @y.setter
    def y(self, value):
        if value is None:
            self._y = self.df._columny[0]
        elif str(value) is int:
            self._y = self.df._columny[value]
        elif type(value) == str:
            self._y = value
        else:
            self._y = self.df._columny[0]
            self.df = self.df.replace_y(value)

    @property
    def xlabel(self):
        return self._xlabel
    
    @xlabel.setter
    def xlabel(self, value):
        self._xlabel = value

    @property
    def ylabel(self):
        return self._ylabel
    
    @ylabel.setter
    def ylabel(self, value):
        self._ylabel = value

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
        self.df = copy.copy(df if df is not None else evaluator.df.train)
        self.noise = noise
        self.x1 = x1
        self.x2 = x2
        self.y = y
        self.xlabel = xlabel
        self.ylabel = ylabel
        if title is not None:
            plt.title(title)
        plt.ylabel(self.ylabel) 
        plt.xlabel(self.xlabel)

    @property
    def x1(self):
        return self._x1
    
    @x1.setter
    def x1(self, value):
        if value is None:
            self._x1 = self.df[self.df._columnx[0]]
            self._xlabel = self.df._columnx[0]
        elif value is int:
            self._x1 = self.df[self.df._columnx[value]]
            self._xlabel = self.df._columnx[value]
        elif type(value) == str:
            self._x1 = self.df[value]
            self._xlabel = value
        else:
            self._x1 = value
            self._xlabel = self.df._columnx[0]

    @property
    def x2(self):
        return self._x2
    
    @x2.setter
    def x2(self, value):
        if value is None:
            self._x2 = self.df[self.df._columnx[1]]
            self._ylabel = self.df._columnx[1]
        elif value is int:
            self._x2 = self.df[self.df._columnx[value]]
            self._ylabel = self.df._columnx[value]
        elif type(value) == str:
            self._x2 = self.df[value]
            self._ylabel = value
        else:
            self._x2 = value
            self._ylabel = self.df._columnx[1]
        
    @property
    def y(self):
        return self._y
    
    @y.setter
    def y(self, value):
        if value is None:
            self._y = self.df._columny[0]
        elif str(value) is int:
            self._y = self.df._columny[value]
        elif type(value) == str:
            self._y = value
        else:
            self._y = self.df._columny[0]
            self.df = self.df.replace_y(value)
        
    @property
    def xlabel(self):
        return self._xlabel
    
    @xlabel.setter
    def xlabel(self, value):
        if value is not None:
            self._xlabel = value

    @property
    def ylabel(self):
        return self._ylabel
    
    @ylabel.setter
    def ylabel(self, value):
        if value is not None:
            self._ylabel = value

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
