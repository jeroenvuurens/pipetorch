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
from .study import Study

class Evaluator:
    def __init__(self, df, *metrics):
        self.df = df
        self.metrics = metrics
        self.reset()   # create a fresh set of results
    
    def clone(self):
        return Evaluator(self.df, self.metrics)
    
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
    
    def study(self, **kwargs):
        return Study.create_study(self, **kwargs)
    
    def _metric_to_name(self, metric):
        try:
            return metric.__name__
        except:
            return metric
    
    def compute_metrics(self, true_y, pred_y, df=None):
        df = df if df is not None else self.df
        true_y = self._inverse_scale_y( df, true_y )
        pred_y = self._inverse_scale_y( df, pred_y )
        pred_y = self._1d(pred_y)
        true_y = self._1d(true_y)
        if len(self.metrics) > 0:
            return {self._metric_to_name(m): m(true_y, pred_y) for m in self.metrics}
        return dict()

    def _trunc_float(self, f):
        if type(f) == float:
            return np.float64(f)
        if type(f) == tuple:
            return repr(tuple([np.float32(v) for v in f]))
        elif type(f) == list:
            return repr([np.float32(v) for v in f])
        elif type(f) == np.ndarray:
            return repr(np.array([np.float32(v) for v in f]))
        else:
            return f
    
    def _dict_to_df(self, *dicts):
        r = []
        for d in dicts:
            d = { key:([value] if isinstance(value, Iterable) else value) for key,value in d.items() }
            r.append( pd.DataFrame(d, index=[0]))
        return pd.concat(r, axis=1)
    
    def reset(self):
        self.results = EvaluatorResults.from_evaluator(self)
        self._current_annotation = None
    
    def _resolve_targets(self, *targets):
        """
        Args:
            *targets: str
                one or more target metrics that are used to find the optimum
                if empty, all recorded metrics are used in the order they were registered (usually loss first)
        
        Returns: [ str ]
            a list of target names
        """
        if len(targets) == 0:
            return self._resolve_targets(*self.metrics)
        targets = [ self._metric_to_name(t) for t in targets ]
        return targets

    def metrics_optimum(self, *targets, direction=None, directions=None, validation=None, test=None, **select):
        """
        Finds the optimal value in a training trial, i.e. where the given `optimize` value over the validation
        set is optimal. The optimize metric must be among the metrics that are cached.
        
        When there are multiple target metrics, ties on the first metric are resolved by the second, etc. 
        
        Args:
            *targets: str
                one or more target metrics that are used to find the optimum
                if empty, all recorded metrics are used in the order they were registered (usually loss first)
            
            direction: str ('minimize')
                'minimize' or 'maximize' to return the resp. lowest or highest score on the validation set
            
            directions: [ str ] ('minimize')
                for multi-target training
            
            validation, test: 
                for internal use, allows recursive calls to resolve finding a multi-target optimum
            
            **select: {}
                Defines the subset of results that is used to find the optimal epoch. 
                When empty, the most recent annotation that was used to record metrics is used to
                define this subset. Alternatively, the key:values in select are used to select the
                results that are considered.
        
        Returns:
            { metric:value }
            A dictionary of values obtained over the cached metrics on the validation set
        """
        
        if validation is None:
            validation = self._select(**select).valid
            if len(select) == 0 and self._current_annotation is not None:
                for key, value in self._current_annotation.items():
                    validation = validation[validation[key] == value]
        if test is None:
            if 'test' in self.results.phase.unique():
                test = self._select(**select).test  
                if len(select) == 0 and self._current_annotation is not None:
                    for key, value in self._current_annotation.items():
                        test = test[test[key] == value]
            else:
                test = validation
        targets = self._resolve_targets(*targets)
        for t in targets:
            assert t in validation.metric.unique(), f'target {t} must be a cached metric'
        if directions is None and direction is None:
            return self.metrics_optimum(*targets, directions=[ 'minimize' if t == 'loss' else 'maximize' for t in targets ], validation=validation, test=test)
        if direction is not None:
            assert directions is None, 'You can only use direction or directions'
            return self.metrics_optimum(*targets, directions=[ direction ], validation=validation, test=test)
        m = validation[validation.metric == targets[0]]
        if directions[0] == 'minimize':
            optimum = m.value.min()
        elif directions[0] == 'maximize':
            optimum = m.value.max()
        else:
            assert False, 'direction must be minimize or maximize'
        if 'epoch' in m.columns: # PyTorch mode, lets find the optimal epoch
            epochs = m.epoch[m.value == optimum]
            test = test[test.epoch.isin(epochs)]
            if len(epochs) > 1 and len(targets) > 1:
                validation = validation[validation.epoch.isin(epochs)]
                return self.metrics_optimum( *target[1:], directions=directions[1:], validation=validation, test=test)
            elif len(epochs) > 1:
                test = test.loc[test.epoch == epochs[0]]
        r = {t:test.value[test.metric == t] for t in targets }
        r = {t:(v.item() if len(v) == 1 else v.iloc[-1].item()) for t, v in r.items()}
        return r
    
    def optimum(self, *targets, direction=None, directions=None, validation=None, test=None, **select):
        """
        Finds the optimal value in a training trial, i.e. where the given `optimize` value over the validation
        set is optimal. The optimize metric must be among the metrics that are cached.
        
        When there are multiple target metrics, ties on the first metric are resolved by the second, etc. 
        
        Args:
            *targets: str
                one or more target metrics that are used to find the optimum
                if empty, all recorded metrics are used in the order they were registered (usually loss first)
            
            direction: str ('minimize')
                'minimize' or 'maximize' to return the resp. lowest or highest score on the validation set
            
            directions: [ str ] ('minimize')
                for multi-target training
            
            validation, test: 
                for internal use, allows recursive calls to resolve finding a multi-target optimum
            
            **select: {}
                Defines the subset of results that is used to find the optimal epoch. 
                When empty, the most recent annotation that was used to record metrics is used to
                define this subset. Alternatively, the key:values in select are used to select the
                results that are considered.
        
        Returns:
            { metric:value }
            A dictionary of values obtained over the cached metrics on the validation set
        """
        targets = self._resolve_targets(*targets)
        r = self.metrics_optimum(*targets, direction=None, directions=None, validation=None, test=None, **select)
        return [ r[t] for t in targets ]
    
    def train_valid_sklearn(self, df):
        pass
#             model.fit(df.train_X, df.train_y)
#             valid_pred = df.valid._predict(predict)
#             metrics = self.compute_metrics(df.valid_y, valid_pred, df)
#             for k, v in metrics.items():
#                 results_valid[k] += v * len(df.valid_X) / len(df)
#             train_pred = dfk.train._predict(predict)
#             y = dfk._inverse_scale_y( y )
#             y_pred = dfk._inverse_scale_y( y_pred )
#             metrics = self.compute_metrics(y, y_pred)
#             for k, v in metrics.items():
#                 results_train[k] += v  * len(dfk.train_X) / len(df)
    
    def run(self, train, predict, model=None, df=None, n_splits=1, stratify=None, **userannot):
        if df is None:
            df = self.df
        if n_splits == 1:
            train(df.train_X, df.train_y)
            self.score_train(predict, df=df, **userannot)
            self.score_valid(predict, df=df, **userannot)
        else:
            train_y = []
            train_y_pred = []
            valid_y = []
            valid_y_pred = []
            for dfk in df.kfold(n_splits=n_splits, stratify=stratify):
                train(dfk.train_X, dfk.train_y)
                y_pred = dfk.valid._predict_y(predict)
                valid_y.append(dfk.valid_y)
                valid_y_predict.append(dfk.valid._predict_y(predict))
                y_pred = dfk.train._predict_y(predict)
                train_y.append(dfk.train_y)
                train_y_predict.append(dfk.train._predict_y(predict))
            self._store_metrics(train_y, train_y_pred, df=dfk, annot={'phase':'train'}, **userannot)
            self._store_metrics(valid_y, valid_y_pred, df=dfk, annot={'phase':'valid'}, **userannot)
             
    def run_sklearn(self, model, df=None, **annot):
        self.run(model.fit, model.predict, model=model, df=df, **annot)

    def _inverse_scale_y(self, df, y):
        if callable(getattr(df, "inverse_scale_y", None)):
            return df.inverse_scale_y( y )
        return y
        
    def _store_predict(self, predict, df, annot={}, **userannot):
        y_pred = df._predict_y(predict)
        self._store_metrics(df.y, y_pred, df=df, annot=annot, **userannot)        
        
    def _store_metrics(self, y, y_pred, df=None, annot={}, **userannot):
        if df is None:
            df = self.df
        metrics = self.compute_metrics(y, y_pred, df=df)
        for m, value in metrics.items():
            self._store_metric(m, value, annot=annot, **userannot)
        return metrics

    def _store_metric(self, metric, value, annot={}, **userannot):
        """
        Stores the metric values in the DataFrame with results. 
        There were some issues the logged (user)annotations and float values.
        The floats are sometimes printed with precision flaws, therefore these are
        converted to np.float32. Also list and tuple values of the annotations cause
        problems, therefore these are converted into Strings. 
        """
        
        self._current_annotation = { key:self._trunc_float(value) for key, value in userannot.items() }
        annot = { key:self._trunc_float(value) for key, value in annot.items() }
        self.results = self.results._add(self._dict_to_df({'metric':metric, 'value': value}, annot, self._current_annotation))
            
    def score_train(self, predict, df=None, annot={}, **userannot):
        if df is None:
            df = self.df
        annot['phase'] = 'train'
        self._store_predict(predict, df.train, annot=annot, **userannot)
            
    def score_valid(self, predict, df=None, annot={}, **userannot):
        if df is None:
            df = self.df
        annot['phase'] = 'valid'
        if len(df.valid_X) > 0:
            self._current_annotation = annot
            self._store_predict(predict, df.valid, annot=annot, **userannot)

    def score_test(self, predict, df=None, annot={}, **userannot):
        if df is None:
            df = self.df
        annot['phase'] = 'test'
        if len(df.test_X) > 0:
            self._current_annotation = annot
            self._store_predict(predict, df.test, annot=annot, **userannot)
          
    def score(self, predict, df=None, annot={}, **userannot):
        self.score_train(predict, df=df, annot=annot, **userannot)
        self.score_valid(predict, df=df, annot=annot, **userannot)
        
    def _order(self, X):
        return X[:, 0].argsort(axis=0)

    def scatter2d_class(self, x1=None, x2=None, y=None, xlabel=None, ylabel=None, title=None, loc='best', noise=0, df=None, **kwargs):
        f = _figure2d(self, x1=x1, x2=x2, y=y, xlabel=xlabel, ylabel=ylabel, title=title, df=df, noise=noise)
        for c in sorted(np.unique(f.graph_y)):
            indices = (c == f.graph_y)
            try:
                c = int(c)
            except: pass
            plt.scatter(f.graph_x1[indices], f.graph_x2[indices], label=c, **kwargs)
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
    
    def plot_boundary(self, predict, levels=[0.5]):
        ax, xx, yy, boundary = self._boundaries(predict)
        ax.contour(xx, yy, boundary, levels=levels)  
        
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
        
    def _figure(self, x=None, y=None, xlabel = None, ylabel = None, sort=False, title=None, interpolate=0, df=None, fig=plt):
        return _figure(self, x=x, y=y, xlabel=xlabel, ylabel=ylabel, title=title, sort=sort, interpolate=interpolate, df=df, fig=fig)
    
    def _legend(self, fig, loc, **legendargs):
        """
        A legend is only drawn is labels are added to the figure.
        This fuctions adds the possibility to use loc=right to put the 
        legend outside the figure for readability.
        """
        if loc == 'right':
            legendargs['bbox_to_anchor'] = (1.04, 1)
            legendargs['loc'] = 'upper left'
            plt.subplots_adjust(right=0.75)
        else:
            legendargs['loc'] = loc
        fig.legend(**legendargs)
    
    def _plot(self, pltfunction, x=None, y=None, xlabel = None, ylabel = None, sort=False, title=None, marker=None, interpolate=0, df=None, loc='best', fig=plt, legendargs={}, **kwargs):
        f = _figure(self, x=x, y=y, xlabel=xlabel, ylabel=ylabel, title=title, sort=sort, interpolate=interpolate, df=df, fig=fig)
        pltfunction(f.graph_x, f.graph_y, marker=marker, **kwargs)
        if 'label' in kwargs:
            self._legend(f.fig, loc, **legendargs)
    
    def line(self, x=None, y=None, xlabel = None, ylabel = None, title=None, interpolate=0, df=None, loc='best', fig=plt, **kwargs):
        plot = fig.plot if fig else plt.plot
        self._plot(plot, x=x, y=y, xlabel=xlabel, ylabel=ylabel, title=title, interpolate=interpolate, sort=True, df=df, loc=loc, fig=fig, **kwargs)

    def scatter(self, x=None, y=None, xlabel = None, ylabel = None, title=None, interpolate=0, df=None, loc='best', fig=plt, **kwargs):
        plot = fig.scatter if fig else plt.scatter
        self._plot(plot, x=x, y=y, xlabel=xlabel, ylabel=ylabel, title=title, interpolate=interpolate, df=df, loc=loc, fig=fig, **kwargs)
       
    def _select(self, select=None, **kwargs):
        if select is None:
            s = self.results
        elif type(select) is pd.core.series.Series:
            s = self.results[select]
        elif type(select) is EvaluatorResults:
            s = select
        elif type(select) is str:
            s = self.results[self.results.phase == select]
        elif type(select) is dict:
            for key, value in select.items():
                s = self.results[self.results[key] == value]
        else:
            raise ValueError('Unknown type passed for select')
        for key, value in kwargs.items():
            s = s.loc[s[key] == value].copy()
        return s

    def _groups(self, selection, series='phase'):
        for g, d in selection.groupby(by=series):
            yield g, self.results._copy_meta(d)
    
    def scatter_metric(self, x, y=None, series='phase', select=None, xlabel = None, ylabel = None, title=None, label_prefix='', label=None, fig=None, **kwargs):
        y = y or self.metrics[0].__name__
        selection = self._select(select)
        selection = selection[selection.metric == y]
        ylabel = ylabel or y
        unique_groups = self._unique(selection, series)
        for g, d in self._groups(selection, series=series):
            g = label or (label_prefix + str(g) if unique_groups > 1 else ylabel)
            d.scatter(x, y='value', xlabel=xlabel, ylabel=y, title=title, label=g, fig=fig, **kwargs)
    
    def line_metric(self, x, y=None, series='phase', select=None, xlabel = None, ylabel = None, title=None, label_prefix='', label=None, fig=None, legendargs={}, **kwargs):
        y = y or self.metrics[0].__name__
        selection = self._select(select)
        selection = selection[selection.metric == y]
        ylabel = ylabel or y
        unique_groups = len([ a for a in self._groups(selection, series=series) ])
        for g, d in self._groups(selection, series=series):
            g = label or (label_prefix + str(g) if unique_groups > 1 else ylabel)
            d.line(x, y='value', xlabel=xlabel, ylabel=ylabel, title=title, label=g, fig=fig, **kwargs)
        
class _figures:
    def _graph_coords_callable(self, df, f):
        if callable(f):
            return self.evaluator.df.inverse_scale_y( f(df.X) ).to_numpy()
        elif type(f) == str:
            return np.squeeze(df[[f]].to_numpy())
        return f
    
    def _graph_coords(self, df, *fields):
        return  [ self._graph_coords_callable(df, f) for f in fields ]         
        
class _figure(_figures):
    def __init__(self, evaluator, x=None, y=None, xlabel = None, ylabel = None, title = None, sort=False, interpolate=0, phase='train', df=None, fig=plt ):
        self.evaluator = evaluator
        self.df = copy.copy(df if df is not None else evaluator.df.train)
        self.x = x
        self.xlabel = xlabel or self.x
        if interpolate > 0:
            assert (y is None) or (type(y)==str) or callable(y), 'You cannot interpolate with given results'
            self.df = self.df.interpolate_factor(interpolate)
            self.y = y
        elif sort:
            self.y = y
            self.df = self.df.sort_values(by=self.x)
        else:
            self.y = y
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
        elif type(value) is int:
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
    def __init__(self, evaluator, x1=None, x2=None, y=None, xlabel = None, ylabel = None, title = None, df=None, noise=0, fig=plt ):
        self.evaluator = evaluator
        self.df = copy.copy(df if df is not None else evaluator.df.train)
        self.noise = noise
        self.x1 = x1
        self.x2 = x2
        self.y = y
        self.xlabel = xlabel
        self.ylabel = ylabel
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
