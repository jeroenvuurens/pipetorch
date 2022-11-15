
import optuna
import joblib
from optuna.storages._cached_storage import _CachedStorage
from optuna.storages._heartbeat import is_heartbeat_enabled
from optuna.trial._state import TrialState
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import RANSACRegressor, HuberRegressor, TheilSenRegressor, LinearRegression
from tqdm.notebook import tqdm
from sklearn.metrics import r2_score
import scipy.stats as st
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import inspect
from collections import defaultdict
from functools import partial
from joblib import parallel_backend
import copy
import random
from optuna.exceptions import ExperimentalWarning
import warnings
from scipy.optimize import minimize

warnings.filterwarnings("ignore", category=ExperimentalWarning, module="optuna")

class Study(optuna.study.Study):
    """
    Extension to an optuna Study. This extension caches the target functions and plot_hyperparameters
    provides a good side-by-side overvew of the hyperparameters over the targets.
    
    For more information, check out Optuna Study.
    """
    
    def __init__(self, study, *target, trainer=None, evaluator=None, grid=None, prune_none=True):
        """
        Call create_study to instantiate a study
        """
        super().__init__(study.study_name, study._storage, study.sampler, study.pruner)
        assert len(target) > 0, 'You need to define at least one target'
        for t in target:
            assert type(t) == str, 'Only str names for targets are currently supported'
        self.target = target
        self.trainer = trainer
        self.evaluator = evaluator
        self.grid = grid
        self._filter_sd = None
        self._filter_upper = None
        self._filter_lower = None
        self._prune_none = prune_none
        
    @classmethod
    def from_study(cls, study, trials=None):
        r = cls.create_study(*study.target, 
                             trainer=study.trainer,
                             evaluator=study.evaluator,
                             storage=study._storage, 
                             sampler=study.sampler,
                             pruner=study.pruner)
        trials = trials or study.trials
        r.add_trials(trials)
        return r
        
    @classmethod
    def create_study(cls, *target, trainer=None, evaluator=None, storage=None, 
                     sampler=None, multivariate=True, pruner=None, 
                     study_name=None, direction=None, 
                     directions=None, load_if_exists=False, grid=None):
        """
        Uses optuna.create_study to create a Study. This extension registers the target metrics for inspection.
        
        Arguments:
            *target: 'loss', str, callable, Trainer or Evaluator
                When called with no targets, this is set to 'loss'
                When called with a trainer, this is set to 'loss' + all metrics that are registered 
                by the trainer.
                When called with an evaluator, this is set all metrics that are registered 
                by the evaluator.
                Otherwise call with a sequence of callables or strings in the same order they are registered by
                the Trainer that is used, e.g. a `Trainer(metrics='f1_score')` will have the `optimum` function return
                `(loss, f1_score)`, therefore, register the study with `Study.create_study('loss', f1_score)`.
                When direction is omitted, loss is set to minimize and all other directions to maximize.
            sampler: optuna Sampler (None)
                By default the multivariate TPE sampler is used. You can override this by passing an
                instantiated Optuna Sampler.
            grid: dict (None)
                If not None, a Grid Search is performed for all possible parameter combinations. This also
                means that you do not have to specify n_trials in optimize, since it will only sample every
                combination once.
                e.g. grid={'lr':[1e-2, 1e-3], 'hidden':range(100, 1000, 100)}
                You cannot combine this with a sampler (since this used GridSampler). 
            trainer: Trainer (None)
                the PipeTorch Trainer this study was constructed with
            evaluator: Evaluator (None) 
                a PipeTorch Evaluator that will be used to store results and to
                compute the evaluation metrics when called with optimum().
                
                When used with a Trainer, every trial will obtain a new Evaluator, 
                unless an evaluator is specified here, this will reuse this evaluator.
                
                Otherwise, the Evaluator is coming from a DFrame.
            other arguments: check optuna
        """
        trainer
        if grid is not None:
            assert type(grid) == dict, 'You have to pass a dict to grid'
            assert sampler is None, 'You cannot use grid together with a custom sampler'
            sampler = optuna.samplers.GridSampler(grid)
        elif sampler is None:
            sampler = optuna.samplers.TPESampler(multivariate=True)
        if len(target) == 0:
            if trainer is not None:
                target = ['loss'] + [ m.__name__ for m in trainer.metrics ]
            elif evaluator is not None:
                target = [ m.__name__ for m in evaluator.metrics ]
            else:
                target = ['loss']
        if direction is None and directions is None:
            if len(target) > 1:
                directions = [ 'minimize' if t == 'loss' else 'maximize' for t in target ]
            else:
                direction = 'minimize' if target[0] == 'loss' else 'maximize'
        study = optuna.create_study(storage=storage, sampler=sampler, pruner=pruner,
                                    study_name=study_name, direction=direction, directions=directions, 
                                    load_if_exists=load_if_exists)
        return cls(study, *target, trainer=trainer, evaluator=evaluator, grid=grid)
    
    @classmethod
    def load(cls, file):
        return joblib.load(file)
    
    def save(self, file):
        oldtrainer = self.trainer
        oldevaluator = self.evaluator
        self.trainer = None
        self.evaluator = None
        joblib.dump(self, file)
        self.trainer = oldtrainer
        self.evaluator = oldevaluator

    def __getitem__(self, slice):
        return self.from_study(self, self.trials[slice])
    
    def sample(self, n):
        """
        Added for checking optimization robustness, by returning
        a study with a random sample of performed trials.
        
        Args:
            n: float or int
                fraction or number of samples to be taken
        
        Returns: Study
            with a sample of the performed trials.
        """
        if type(n) == float:
            n = int(round(len(self.trials) * n))
        with self.quiet_mode():
            r = self.from_study(self, random.sample(self.trials, n))
        return r
    
    def constraint(self, **kwargs):
        """
        Return a copy of the study that only contain the trails for which the
        hyperparameters are within the given ranges. 
        
        Example:
            constraint(lr=[0.01, 0.1], hidden=[100, 400])
        
        Args:
            **kwargs: dict
                In kwargs, pass the parameter names with a range [min, max]. 
                The min and max boundaries are inclusive. 
        
        Returns: Study
            with only the trials that meet the given constraints
        """
        r = self.results
        for p, (minx, maxx) in kwargs.items():
            rp = r[(r.parameter == p) & (r.parametersetting >= minx) 
                                      & (r.parametersetting <= maxx)]
            trials = set(rp.trial.unique())
            r = r[r.trial.isin(trials)]
        trials = [ t for t in self.trials if t.number in trials ]
        return Study.from_study(self, trials)
    
    def ask(self, fixed_distributions=None):
        if not self._optimize_lock.locked():
            if is_heartbeat_enabled(self._storage):
                warnings.warn("Heartbeat of storage is supposed to be used with Study.optimize.")

        fixed_distributions = fixed_distributions or {}
        fixed_distributions = {
            key: _convert_old_distribution_to_new_distribution(dist)
            for key, dist in fixed_distributions.items()
        }

        # Sync storage once every trial.
        if isinstance(self._storage, _CachedStorage):
            self._storage.read_trials_from_remote_storage(self._study_id)

        trial_id = self._pop_waiting_trial_id()
        if trial_id is None:
            trial_id = self._storage.create_new_trial(self._study_id)
        trial = Trial(self, trial_id)

        for name, param in fixed_distributions.items():
            trial._suggest(name, param)

        return trial

    def quiet_mode(self, quiet=True):
        """
        A ContextManager to silence optuna
        """
        class CM(object):
            def __enter__(self):
                if quiet:
                    self.old_verbosity = optuna.logging.get_verbosity()
                    optuna.logging.set_verbosity(optuna.logging.ERROR)
            
            def __exit__(self, type, value, traceback):
                if quiet:
                    optuna.logging.set_verbosity(self.old_verbosity)

        return CM()
    
    def optimize(self, func, n_trials=None, timeout=None, catch=(), callbacks=None, 
                 gc_after_trial=True, show_progress_bar=False, n_jobs=1):
        """
        See Optuna's optimize, this extensions adds passing the trainer to the trial function.
        """
        
        args = len(inspect.getfullargspec(func)[0])
        if args == 2:
            if self.trainer is not None:
                func = partial(func, self.trainer)
            elif self.evaluator is not None:
                func = partial(func, self.evaluator)
            else:
                raise NameError('You can only pass a func with two arguments when trainer/evaluator is set')
        try:
            del self._rules
        except: pass
        try:
            del self._results
        except: pass
        with self.quiet_mode(show_progress_bar):
            super().optimize(func, n_trials=n_trials, 
                             timeout=timeout, catch=catch, 
                             callbacks=callbacks,
                             gc_after_trial=gc_after_trial, 
                             show_progress_bar=show_progress_bar, n_jobs=n_jobs)        

    def add_trial(self, trial):
        if trial.state.name == 'COMPLETE':
            super().add_trial(trial)
        try:
            del self._results
        except: pass
        try:
            del self._rules
        except: pass

    @property
    def trials(self):
        return super().get_trials(deepcopy=True, states=[ TrialState.COMPLETE ] )
    
    @property
    def parameters(self):
        return self.trials[0].params.keys()

    @property
    def results(self):
        try:
            return self._results
        except:
            table = []
            for t in self.trials:
                for param, paramv in t.params.items():
                    for target, value in zip(self.target, t.values):
                        table.append((t.number, param, paramv, target, value))
            if len(table) > 0:
                return pd.DataFrame(table, columns=['trial', 'parameter', 'parametersetting', 
                                                    'target', 'targetvalue'])
            for t in self.trials:
                for target, value in zip(self.target, t.values):
                    table.append((t.number, target, value))
            self._results = pd.DataFrame(table, columns=['trial', 'target', 'targetvalue'])
            return self._results
   
    def target_direction(self, target='loss'):
        """
        Args:
            target: str ('loss')
                The target to return the direction for
        
        Return: bool
            True if minimize, else False
        """
        targeti = self.target.index(target)
        return self.directions[targeti].value == 1

    def selected_results(self, parameter, target):
        return self.results[(self.results.parameter == parameter) & (self.results.target == target)]

    def _filtered_target(self):
        r = set(self.results.trial)
        df = self.results[self.results.trial.isin(self.filtered_trials())]
        if self._filter_sd is not None:
            firstp = list(self.parameters)[0]
            s = (df.parameter == firstp) 
            s = s & (df.target == self._filter_target)
            t = df[s].targetvalue
            mean = np.mean(t)
            sd = np.std(t)
            s = set(df[(df.targetvalue > mean - self._filter_sd * sd) & (df.targetvalue < mean + self._filter_sd * sd)].trial)
            r = r.intersection(s)
        if self._filter_lower is not None:
            s = (df.targetvalue > self._filter_lower) 
            s = s & (df.target == self._filter_target)
            s = set(df[s].trial)
            r = r.intersection(s)
        if self._filter_upper is not None:
            s = (df.targetvalue < self._filter_upper) 
            s = s & (df.target == self._filter_target)
            s = set(df[s].trial)
            r = r.intersection(s)
        return r
    
    def filter_target(self, sd=None, lower=None, upper=None, target=None):
        """
        Filter the results so that points are excluded based on the following rules:
        
        Args:
            sd: float (None)
                when set, compute the mean and std for the target and exclude 
                all results outside 
                [ mean(target) - sd * std(target), mean(target) + sd * std(target)]
                
            lower: float (None)
                when set, exclude points with a targetvalue lower that the given threshold
                
            higher: float (None)
                when set, exclude points with a targetvalue higher that the given threshold
        """
        if target is None:
            target = self.target[1] if len(self.target) > 1 else self.target[0]
        self._filter_sd = sd
        self._filter_lower = lower
        self._filter_upper = upper
        self._filter_target = target
    
    def filter_upper(self, target=None):
        if target is None:
            target = self.target[1] if len(self.target) > 1 else self.target[0]
        firstp = list(self.parameters)[0]
        s = (self.results.parameter == firstp) 
        s = s & (self.results.target == target)
        t = self.results[s].targetvalue
        mean = np.mean(t)
        self.filter_target(upper=mean, target=target)
    
    def filter_lower(self, target=None):
        if target is None:
            target = self.target[1] if len(self.target) > 1 else self.target[0]
        firstp = list(self.parameters)[0]
        s = (self.results.parameter == firstp) 
        s = s & (self.results.target == target)
        t = self.results[s].targetvalue
        mean = np.mean(t)
        self.filter_target(lower=mean, target=target)
    
    @property
    def rules(self):
        try:
            return self._rules
        except:
            if len(self.trials) > 0:
                #t = self._filtered_target()
                self._rules = pd.DataFrame(columns=['parameter', 'low', 'high'])
                t = self.trials[0]
                for parameter, dist in t.distributions.items():
                    try:
                        self._rules.loc[len(self._rules)] = (parameter, None, None)
                    except: pass
                return self._rules
            
    def filtered_trials(self):
        df = self.results
        s = set(df.trial)
        for i, r in self.rules.iterrows():
            f = (df.parameter == r.parameter)
            if r.low is not None:
                f = f & (df.parametersetting  > r.low)
            if r.high is not None:
                f = f & (df.parametersetting  < r.high)
            s = s.intersection(set(df[f].trial))
        return s
    
    def filtered_results(self):
        df = self.results
        df = df[df.trial.isin(self._filtered_target())]
        df = df[df.trial.isin(self.filtered_trials())]  
        return df

    def pivotted_results(self, target=None):
        from pipetorch.data import DFrame
        target = target or self.default_target()
        targetvalues = self.results[self.results.target == target]
        targetvalues = targetvalues[['trial', 'targetvalue']].drop_duplicates().set_index('trial')
        trials = self.results.pivot_table(index='trial', columns='parameter', values='parametersetting', aggfunc = np.mean)
        data = trials.join(targetvalues)
        return DFrame(data)
    
    def rule(self, parameter, low=None, high=None):
        if low is not None:
            self.rules.loc[self.rules.parameter == parameter, 'low'] = low
        if high is not None:
            self.rules.loc[self.rules.parameter == parameter, 'high'] = high   

    def reset_rules(self):
        try:
            del self._rules
        except: pass
            
    def distribution(self, param):
        dist = self.trials[0].distributions[param]
        return dist
    
    def is_log_distribution(self, param):
        return self.distribution(param).log

    def tune_r2_repeated(self, n=10, sample=0.8, **kwargs):
        results = []
        for i in range(n):
            s = self.sample(sample)
            r = s.tune_r2(**kwargs)
            r = r.pivot_table(columns='parameter').reset_index(drop=True)
            results.append(r)
        return pd.concat(results)
        
    def sort2(self, a, b):
        return (a, b) if a < b else (b, a)

    def default_target(self):
        if len(self.target) > 1 and self.target[0] == 'loss':
            return self.target[1]
        return self.target[0]

    def tune_ransac(self, *parameters, degree=3, target=None, best=20, minimum=5, **kwargs):
        results = self._tune_ransac(*parameters, degree=degree, target=target, best=best, minimum=minimum, **kwargs)
        return self.tune(results=results, target=target, parameters=parameters, best=best, minimum=minimum)
    
    def _tune_ransac(self, *parameters, degree=3, target=None, best=20, minimum=5, **kwargs):
        parameters = parameters if len(parameters) > 0 else self.parameters
        target = target or self.default_target()
        r = self.pivotted_results(target)
        r = r.columnx(*parameters).columny('targetvalue')
        r = r.polynomials(degree=degree)
        model = RANSACRegressor(**kwargs)
        for c in self.parameters:
            if self.is_log_distribution(c):
                r = r.log(c)
        model.fit(r.train_X, r.train_y)
        r['targetvalue'] = model.predict(r.train_X)
        r = r.reset_index()
        r = pd.melt(r, value_vars=self.parameters, id_vars=['targetvalue', 'trial'], 
                    var_name='parameter', value_name='parametersetting')
        r['target'] = target
        return pd.DataFrame(r)
 
    def tunej(self, *parameters, degree=3, target=None, best=20, minimum=5, factor=5, **kwargs):
        results = self._tunej(*parameters, degree=degree, target=target, best=best, minimum=minimum, factor=factor, **kwargs)
        return self.tune(results=results, target=target, parameters=parameters, best=best, minimum=minimum)
    
    def _tunej(self, *parameters, degree=3, target=None, best=20, minimum=5, factor=5, **kwargs):
        parameters = parameters if len(parameters) > 0 else self.parameters
        target = target or self.default_target()
        r = self.pivotted_results(target)
        r = r.columnx(*parameters).columny('targetvalue')
        r = r.polynomials(degree=degree)
        for c in self.parameters:
            if self.is_log_distribution(c):
                r = r.log(c)
        weights = None
        for i in range(10):
            model = LinearRegression()
            model.fit(r.train_X, r.train_y, sample_weight=weights)
            y_pred = model.predict(r.train_X)
            mean_error = np.mean(r.train_y - y_pred)
            std_error = np.std(r.train_y - y_pred) / factor
            z_error = (abs(r.train_y - y_pred) - mean_error) / std_error
            weights = 1 - st.norm.cdf(z_error)

        r['targetvalue'] = model.predict(r.train_X)
        r = r.reset_index()
        r = pd.melt(r, value_vars=self.parameters, id_vars=['targetvalue', 'trial'], 
                    var_name='parameter', value_name='parametersetting')
        r['target'] = target
        return pd.DataFrame(r)

    def tune(self, results=None, target=None, parameters=None, best=20, minimum=5):
        """
        Finds the optimal hyperparameter settings by sampling intervals
        between combinations of the #best trials.
        
        The best hyperparameter settings are found by sampling combinations of
        relatively good trials. Every combination sets the boundaries for the
        hyperparameters by the minimum and maximum values of these points, thus
        selecting all trials that fall in between these boundaries. Combinations 
        that fail to include at least the set 'minimum' of points are dismissed.
        
        Arguments:
            target: str (None)
                the target (objective) that is used for the optimization. When None,
                by default the first objective of the study after the loss is used. 
                If there is no first objective, then loss is used.
            parameters: [ str ] (None)
                the hyperparameters to be tuned. If None, all are tuned simultaneously.
                This can be used to focus on the most sensitive parameters only.
            best: int (10)
                uses the #best trials to sample combinations from
            minimum: int(5)
                only combinations that include at least #minimum trials are considered.
                when no combination satisfied the minimum constraint (which sometimes
                happens), the combination with the highest number of trials are used.
        
        Returns: DataFrame
            the columns in the DataFrame correspond to this studies' hyperparameters
            and the row contains an estimation of the optimal values for each.
        """
        target = target or self.default_target()
        parameters = parameters or self.parameters
        results = results if results is not None else self.results
        assert all([ (p in self.parameters) for p in parameters ]), \
            "all parameters should be parameters that are tuned in this study"
        assert target in self.target, 'target should be an objective in this study'
        
        direction = self.target_direction(target)
        r = results[results.target == target]

        paramsetting = {}
        paramtrial = {}
        paramindex = {}
        for p in parameters:
            rp = r[r.parameter == p].sort_values(by='parametersetting')
            paramsetting[p] = rp.parametersetting.to_list()
            paramtrial[p] = rp.trial.to_list()
            paramindex[p] = { t:i for i, t in enumerate(paramtrial[p]) }
        trial2targetvalue = { t.trial:t.targetvalue for t in rp.itertuples() }
        pivots = rp.sort_values(by='targetvalue', ascending=(direction==1))
        alltrials = set(rp.trial.unique())       
        maxtargetvalue = None
        maxtrials = None
        maxminimum = 0
        best = min(best, len(pivots))
        while maxtrials is None:
            for i in range(best-1):
                for j in range(i+1, best):
                    currenttrials = set(alltrials)
                    triali = pivots.iloc[i].trial
                    trialj = pivots.iloc[j].trial
                    for p in parameters:
                        mini, maxi = self.sort2(paramindex[p][triali], paramindex[p][trialj])
                        currenttrials = currenttrials.intersection(set(paramtrial[p][mini:maxi+1]))
                    if len(currenttrials) >= minimum:
                        targetvalue = sum([ trial2targetvalue[t] for t in currenttrials ])/len(currenttrials)
                        if maxtargetvalue is None or \
                           (direction == 1 and targetvalue < maxtargetvalue) or \
                           (direction == 0 and targetvalue > maxtargetvalue):
                            maxtargetvalue = targetvalue
                            maxtrials = currenttrials
                    else:
                        maxminimum = max(len(currenttrials), maxminimum)
            if maxtrials is None:
                minimum = maxminimum

        r = r[r.trial.isin(maxtrials)]
        trials = r.pivot_table(index='trial', columns='parameter', values='parametersetting')
        result = self.median_parametersettings(trials)
        for t in self.target:
            r = results[(self.results.target == t)]
            r = r[r.trial.isin(maxtrials)]
            result[t] = r.targetvalue.mean()
        return result


    def tune_repeated(self, n=10, sample=0.8, **kwargs):
        """
        Tunes the hyperparameters repeatedly to inspect the sensitivity of the
        hyperparameter tuning.
        
        Args:
            n: int (10)
                number of times to repeat the tuning
            sample: float (0.8)
                a new random sample of this fraction is used for each iteration
            **kwargs: arguments passed to tune()
            
        Returns: DataFrame
            The columns correspond to this studies hyperparameters and in the rows
            are the results for the repeated tests
        """
        results = []
        for i in tqdm(range(n)):
            s = self.sample(sample)
            r = s.tune(**kwargs)
            results.append(r)
        return pd.concat(results)
    
    def median_parametersettings(self, df):
        """
        Correctly computes the parameter median from a DataFrame with
        parameters settings, taking into account that some parameters
        may be on a log scale.
        
        Args: DataFrame
            The columns should correspond to this studies hyperparameters
            and the rows are the values to be averaged.
            
        Returns: DataFrame
            With the parameter medians.
        """
        m = {}
        for p in df.columns:
            if p in self.parameters and self.is_log_distribution(p):
                m[p] = np.log(np.median(np.exp(df[p])))
            else:
                m[p] = df[p].median()
        return pd.DataFrame([m])

    def tune_bootstrap(self, n=10, sample=0.8, **kwargs):
        """
        Performs a bootstrap estimation of the hyperparameter tuning.
        
        Args:
            n: int (10)
                number of times to repeat the tuning
            sample: float (0.8)
                a new random sample of this fraction is used for each iteration
            **kwargs: arguments passed to tune()
            
        Returns: DataFrame
            the columns in the DataFrame correspond to this studies' hyperparameters
            and the row contains an estimation of the optimal values for each.
        """

        r = self.tune_repeated(n=n, sample=sample, **kwargs)
        return self.median_parametersettings(r)
        
    def tune_r2(self, keep=0.3, target=None, min_r2=0, degree=2, alpha=1e-4):
        target = target or self.default_target()
        total = len(self.filtered_results())/len(self.parameters)/len(self.target)
        if type(keep) == float:
            keep = keep * total
        pbar = tqdm(total=total - keep)
        while total > keep:
            opt = [ Optimum(self, p, target, degree=degree, alpha=alpha) 
                    for p in self.parameters ]
            opt = [ (o, *o.predict_left_right()) for o in opt ]
            left = sorted([ (l, o) for o, l, r in opt ])
            right = sorted([ (r, o) for o, l, r in opt ])
            old_total = total
            if self.target_direction(target):
                lr, l = left[-1]
                rr, r = right[-1]
                if lr > rr:
                    self.rule(l.parameter, low=l.df.parametersetting.min())
                else:
                    self.rule(r.parameter, high=r.df.parametersetting.max())
            else:
                lr, l = left[0]
                rr, r = right[0]
                if rr > lr:
                    self.rule(l.parameter, low=l.df.parametersetting.min())
                else:
                    self.rule(r.parameter, high=r.df.parametersetting.max())
            total = len(self.filtered_results())/len(self.parameters)/len(self.target)
            pbar.update(old_total - total)
        pbar.close()
        r = self.filtered_results()
        r = r[r.target == target]
        return r.groupby(by='parameter').parametersetting.mean().to_frame()

    def plot_hyperparameter(self, parameter, target=None, ax=plt, fit=None, show='out', **modelparameters):
        """
        Plot a scatter graph for a specific parameter and target
        
        Args:
            parameter: str or int
                specifies the parameter by name or position in the parameter list
            target: str or int
                specifies the target (objective function) by name or
                position in the target list
            ax: plt Axis (plt)
                when specified this should be the axis object of a subfigure to
                create side-by-side plots
            fit: bool (None)
                if True, a polynomial regression line will be fitted and shown on the data
            degree: int (2)
                the degree of the polynomial regression function to be fitted
            alpha: float (1e-4)
                the regularization parameter of the regression function to be fitted
            show: str
                'in' will show only the datapoints that are filtered in
                'out' will show the datapoints that were filtered out as tiny dots
                'all' will disregard the current filter and show all points
        """
        self.scatter_hyperparameter(parameter, target, ax, fit, show, **modelparameters)
    
    def scatter_hyperparameter(self, parameter=0, target=None, ax=plt, fit=None, show='out', figsize=(3,3), **modelparameters):
        """
        Plot a scatter graph for a specific parameter and target
        
        Args:
            parameter: str or int (0)
                specifies the parameter by name or position in the parameter list
            target: str or int (None)
                specifies the target (objective function) by name or
                position in the target list, default is the first objective
                besides the loss function.
            ax: plt Axis (plt)
                when specified this should be the axis object of a subfigure to
                create side-by-side plots
            fit: bool (None)
                if True, a polynomial regression line will be fitted and shown on the data
            degree: int (2)
                the degree of the polynomial regression function to be fitted
            alpha: float (1e-4)
                the regularization parameter of the regression function to be fitted
            show: str
                'in' will show only the datapoints that are filtered in
                'out' will show the datapoints that were filtered out as tiny dots
                'all' will disregard the current filter and show all points
        """
        if type(parameter) == int:
            parameter = self.parameters[parami]
        if ax == plt:
            plt.figure(figsize=figsize)
        if target is None:
            target = self.default_target()
        elif type(target) == int:
            target = self.target[targeti]
        if fit:
            if fit=='RANSAC':
                o = OptimumRANSAC(self, parameter, target, **modelparameters)
            elif fit=='Huber':
                o = OptimumHuber(self, parameter, target, **modelparameters)
            elif fit=='TheilSen':
                o = OptimumTheilSen(self, parameter, target, **modelparameters)
            elif fit=='J':
                o = OptimumJ(self, parameter, target, **modelparameters)
            else:
                o = Optimum(self, parameter, target, **modelparameters)
            #if o.r2 > 0:
            o.plot(ax)
        ylim = plt.ylim if ax == plt else ax.set_ylim
        title = plt.title if ax == plt else ax.set_title
        xscale = plt.xscale if ax == plt else ax.set_xscale
        ylabel = plt.ylabel if ax == plt else ax.set_ylabel
        xlabel = plt.xlabel if ax == plt else ax.set_xlabel
        
        subset = self.selected_results(parameter, target)
        if show == 'in' or show == 'out':
            trials = self.filtered_trials().intersection(self._filtered_target())
            subset_in = subset[subset.trial.isin(trials)]
            self._scatter(ax, subset_in)
            if show == 'out':
                self._scatter_hidden(ax, subset[~subset.trial.isin(trials)])
            ylim(subset_in.targetvalue.min(), subset_in.targetvalue.max())
        else:
            self._scatter(ax, subset)
            ylim(subset.targetvalue.min(), subset.targetvalue.max())
            
        title(parameter)
        if self.is_log_distribution(parameter):
            xscale('log')
        ylabel(target)
    
    def scatter_hyperparameters(self, figsize=None, logscale=['loss'], fit=True, show='out', **modelparameters):
        """
        Plots scatter graps of each hyperparameter over each recorded metric to 
        allow manual optimization and provide insight to the sensitivity.
        
        Arguments:
            figsize: (width, height) None
                controls the size of the figure displayed
            logscale: ['loss']
                list of metrics whose y-axis is shown as a log scale. By default this is done for the loss
                because the learning rate is often sampled from a log distribution and this makes it easier
                to estimate the optimum.
            fit, degree, alpha, show: see scatter_hyperparameter
        """
        results = self.results
        parameters = self.parameters
        if figsize is None:
            figsize = (4 * len(parameters), 4 * len(self.target))
        
        fig, axs = plt.subplots(len(self.target), len(parameters), sharex='col', sharey='row', figsize=figsize)
        
        if len(parameters) == 1:
            if len(self.target) == 1:
                axs = np.array([[axs]])
            else:
                axs = np.expand_dims(axs, axis=1)
        elif len(self.target) == 1:
            axs = np.expand_dims(axs, axis=0)
        for parami, param in enumerate(parameters):
            for targeti, target in enumerate(self.target):
                ax = axs[targeti, parami]
                self.scatter_hyperparameter(param, target, ax=ax, fit=fit, show=show, **modelparameters)
                if target in logscale:
                    ax.set_yscale('log')

    def plot_hyperparameters(self, figsize=None, logscale=['loss'], fit=False, **modelparameters):
        """
        Plots scatter graps of each hyperparameter over each recorded metric to 
        allow manual optimization and provide insight to the sensitivity.
        
        Arguments:
            figsize: (width, height) None
                controls the size of the figure displayed
            logscale: ['loss']
                list of metrics whose y-axis is shown as a log scale. By default this is done for the loss
                because the learning rate is often sampled from a log distribution and this makes it easier
                to estimate the optimum.
            fit and **kwargs: see scatter_hyperparameter.
        """
        self.scatter_hyperparameters(figsize=figsize, logscale=logscale, fit=fit, **modelparameters)
                    
    def plot_targets(self, *targets, parameter=None, **kwargs):
        """
        Plots the target results in a single figure
        
        Arguments:
            *targets: str (None)
                the targets to plot
            parameter: str (None)
                the parameter to plot, or None to plot all data which is fine when there is only one parameter
            **kwargs: dict
                arguments for Pandas DataFrame.plot
                
        Returns: matplotlib.axes.Axes
            of the plotted figure, which can be used to extend or modify the figure
        """
        r = self.results
        if parameter is not None:
            r = r[r.parameter == parameter]
        targets = targets if len(targets) > 0 else self.target
        
        
        curves = [ r[r.target==t].sort_values(by='parametersetting') for t in targets ]
        for t, c in zip(self.target, curves):
            try:
                c.plot(x='parametersetting', y='targetvalue', label=t, ax=ax)
            except:
                ax = c.plot(x='parametersetting', y='targetvalue', label=t, **kwargs)
        plt.legend()
        return ax
                    
    def trial_targets(self):
        """
        lists to metrics over the trials.
        """
        l = defaultdict(list)
        for t in self.trials:
            for target, value in zip(self.target, t.values):
                l[target].append(value)
        return pd.DataFrame.from_dict(l)        
       
    def validate(self):
        """
        Reports the mean and variance for each metric over the trials, providing more stable outcomes using
        n-fold cross validation.
        """
        l = defaultdict(list)
        for t in self.trials:
            for target, value in zip(self.target, t.values):
                l[target].append(value)
        mean = []
        std = []
        for target, values in l.items():
            mean.append(np.mean(values))
            std.append(np.std(values))
        return pd.DataFrame({'target':self.target, 'mean':mean, 'std':std})        
        
    def _scatter(self, ax, subset):
        x = subset.parametersetting.astype(np.float64)
        y = subset.targetvalue.astype(np.float64)
        z = subset.trial
        ax.scatter(x, y, c=z, cmap='plasma')
        
    def _scatter_hidden(self, ax, subset):
        x = subset.parametersetting.astype(np.float64)
        y = subset.targetvalue.astype(np.float64)
        ax.scatter(x, y, s=1)
            
class Trial(optuna.trial.Trial):
    @property
    def trainer(self):
        """
        A reference to the PipeTorch Trainer object this Study was instantiated through.
        """
        try:
            return self._trainer
        except:
            self._trainer = self.study.trainer
            self._trainer.reset_evaluator()
            self._trainer._evaluator = self.evaluator
            return self._trainer

    @property
    def evaluator(self):
        """
        A reference to the PipeTorch Evaluator object for this Study.
        
        This only works if the Study was instantiated through a PipeTorch
        Trainer or DFrame.
        """
        try:
            return self._evaluator
        except:
            if self.study.evaluator is None:
                self._evaluator = self.trainer.evaluator
            else:
                self._evaluator = self.study.evaluator
            return self._evaluator

    @property
    def df(self):
        """
        A reference to the PipeTorch DFrame for this Study.
        
        This only works if the Study was instantiated through a PipeTorch
        Trainer or DFrame.
        """
        if self.trainer is not None and self.trainer.databunch is not None:
            return self.trainer.databunch.df
        if self.evaluator is not None:
            return self.evaluator.df
        raise NameError('This study was not initialized from a PipeTorch Trainer/DFrame/Evaluator')
    
    @property
    def train_X(self):
        """
        Depending on the source, train_X as a Numpy array (when issued from a DFrame)
        or train_X as a PyTorch Tensor (when issued from a Trainer with a Databunch).
        """
        if self.trainer is not None:
            if self.trainer.databunch is not None:
                return self.trainer.databunch.train_X
            else:
                raise NameError('train_X is currently only supported for Trainers with a DataBunch')
        elif self.df:
            return self.df.train_X
        
    @property
    def valid_X(self):
        """
        Depending on the source, valid_X as a Numpy array (when issued from a DFrame)
        or valid_X as a PyTorch Tensor (when issued from a Trainer with a Databunch.
        """
        if self.trainer is not None:
            if self.trainer.databunch is not None:
                return self.trainer.databunch.valid_X
            else:
                raise NameError('valid_X is currently only supported for Trainers with a DataBunch')
        elif self.df:
            return self.df.valid_X
        
    @property
    def train_y(self):
        """
        Depending on the source, train_y as a Numpy array (when issued from a DFrame)
        or train_y as a PyTorch Tensor (when issued from a Trainer with a Databunch.
        """
        if self.trainer is not None:
            if self.trainer.databunch is not None:
                return self.trainer.databunch.train_y
            else:
                raise NameError('train_y is currently only supported for Trainers with a DataBunch')
        elif self.df:
            return self.df.train_y
        
    @property
    def valid_y(self):
        """
        Depending on the source, valid_y as a Numpy array (when issued from a DFrame)
        or valid_y as a PyTorch Tensor (when issued from a Trainer with a Databunch.
        """
        if self.trainer is not None:
            if self.trainer.databunch is not None:
                return self.trainer.databunch.valid_y
            else:
                raise NameError('valid_y is currently only supported for Trainers with a DataBunch')
        elif self.df:
            return self.df.valid_y
        
    @property
    def train_dl(self):
        """
        Dataloader for the training set, obtained from the PipeTorch Trainer.
        """
        try:
            return self.trainer.train_dl
        except:
            raise NameError('train_dl is currently only supported by Trainers')    
        
    @property
    def valid_dl(self):
        """
        Dataloader for the validation set, obtained from the PipeTorch Trainer.
        """
        try:
            return self.trainer.valid_dl
        except:
            raise NameError('valid_dl is currently only supported by Trainers')    
    
    def optimum(self, *target, direction=None, directions=None, **select):
        """
        Returns the values for the given targets in this trial. This method
        depends on proper use of the PipeTorch Evaluator to records evaluation metrics
        during the trial. The PipeTorch Trainer will do this by default, as optimum
        will return a 'soft optimum' by picking the epoch with the lowest validation loss.
        Alternatively, you can access the evaluator through the trial object to add
        score_train() and score_valid() to the results.
        
        Args:
            *target: str or callable ('loss')
                names or metric functions that are used to decide what training cycle the model was most optimal

            direction: str or [ str ] (None)
                for every target: 'minimize' or 'maximize' to find the highest or lowest value on the given target
                If None, 'minimize' is used when optimize is 'loss', otherwise 'maximize' is used
 
            directions: [ str ] (None)
                same as direction, but now a list of 'minimize' or 'maximize' for multipe targets.
            
            select: {} (None)
                When None, the log={} values from the last call to train() are used. Otherwise, select
                is a dictionary with values that distinguish the results from the current trial 
                to the previous trails, which is needed to find the single best epoch of the current trail
                to return the metrics for that epoch.
                
        Returns:
            [ target ]
            A list of target values 
        """
        if self.trainer is not None:
            return self.trainer.optimum(*target, direction=direction, directions=directions, **select)
        elif self.evaluator is not None:
            return self.evaluator.optimum(*target, direction=direction, directions=directions, **select)
    
    def suggest_categorical(self, name, choices=None):
        try:
            choices = choices or self.study.grid[name]
        except: pass
        return super().suggest_categorical(name, choices)

class Ridge:
    def __init__(self, alpha):
        self.alpha = alpha
        
    def fit(self, X, y):
        leftmat = np.linalg.pinv(X.T @ X + self.alpha * np.identity(X.shape[1]))
        self.betas = leftmat @ X.T @ y
    
    def predict(self, X):
        return X @ self.betas
    
    def optima(self):
        if len(self.betas) == 3:
            _, b, a = [ i * b for i, b in enumerate(self.betas) ]
            return [ -b / (2 * a) ]
        elif len(self.betas) == 4:
            _, c, b, a = [ i * b for i, b in enumerate(self.betas) ]
            if a != 0:
                D = b * b - 4 * a * c
                if D == 0:
                    return [ -b / (2 * a) ]
                elif D > 0:
                    return [ (-b - math.sqrt(D)) / (2 * a), (-b + math.sqrt(D)) / (2 * a) ]
            else:
                return [ -c / (2 * b) ]
        return []
    
class Optimum:
    def __init__(self, study, parameter, target, fold=None, degree=2, alpha=1e-3):
        self.study = study
        self.parameter = parameter
        self.target = target
        self.degree = degree
        self.alpha = alpha
        self.fold = fold
        
    def __repr__(self):
        return f'{self.parameter} {self.target} {self.r2}'
        
    @classmethod
    def loo(cls, study, parameter, target, **kwargs):
        y_true = []
        y_pred = []
        base = cls(study, parameter, target, **kwargs)
        for i in range(1, len(base.df)-1):
            print(parameter, i)
            s = cls(study, parameter, target, fold=i, **kwargs)
            y_pred.append(s.pred_y.item())
            y_true.append(s.valid_y.item())
        return r2_score(y_true, y_pred)
        
    @property
    def is_log_distribution(self):
        return self.study.is_log_distribution(self.parameter)
    
    @property
    def df(self):
        try:
            return self._df
        except:
            self._df = copy.copy(self.study.filtered_results())
            self._df = self._df[(self._df.parameter == self.parameter) & (self._df.target == self.target)]
            self._df = self._df[['parametersetting', 'targetvalue']]
            self._df = self._df.sort_values(by='parametersetting').reset_index(drop=True)
            return self._df
    
    @property
    def train(self):
        if self.fold:
            return self.df.drop(self.fold)
        return self.df
    
    @property
    def valid(self):
        if self.fold:
            return self.df.iloc[self.fold:self.fold+1]
        return self.df
    
    def transform_X(self, X):
        if self.is_log_distribution:
            X = np.log( X )
        p = PolynomialFeatures(degree=self.degree, include_bias=True)
        return p.fit_transform(X)
    
    @property
    def train_X(self):
        return self.transform_X(self.train.loc[:, ['parametersetting']])
    
    @property
    def valid_X(self):
        return self.transform_X(self.valid.loc[:, ['parametersetting']])
    
    @property
    def train_y(self):
        return self.train['targetvalue']
    
    @property
    def valid_y(self):
        return self.valid['targetvalue']
    
    @property
    def minx(self):
        try:
            return self._minx
        except:
            self._minx = self.df.parametersetting.min()
            return self._minx

    @property
    def maxx(self):
        try:
            return self._maxx
        except:
            self._maxx = self.df.parametersetting.max()
            return self._maxx
    
    @property
    def model(self):
        try:
            return self._model
        except:
            self._model = Ridge(alpha=self.alpha)
            self._model.fit(self.train_X, self.train_y)
            return self._model
        
    @property
    def pred_y(self):
        return self.model.predict(self.valid_X)

    def plot(self, ax):
        if self.is_log_distribution:
            X = np.logspace(np.log(self.minx)/np.log(10), np.log(self.maxx)/np.log(10))
        else:
            X = np.linspace(self.minx, self.maxx)
        pred_y = self.model.predict(self.transform_X(X.reshape(-1,1)))
        ax.plot(X, pred_y)

    @property
    def direction(self):
        return self.study.target_direction(self.target)
        
    def predict_left_right(self):
        x = np.array([self.minx, self.maxx])
        y = self.model.predict(self.transform_X(x.reshape(-1,1)))
        y = y.reshape(-1)
        return y
        
    def optima(self):
        try:
            return self._optima
        except:
            inrange = lambda x: x >= self.minx and x <= self.maxx
            x = np.array([self.minx, self.maxx, 
                          *[ x for x in self.model.optima() if inrange(x) ]])
            y = self.model.predict(self.transform_X(x.reshape(-1,1)))
            y = y.reshape(-1)
            if self.direction == 1:
                i = np.argmin(y)
                self._optima = x[i], y[i]
            else:
                i = np.argmax(y)
                self._optima = x[i], y[i]
            return self._optima

    def least_optimal(self):
        try:
            return self._least_optimal
        except:
            inrange = lambda x: x >= self.minx and x <= self.maxx
            x = np.array([self.minx, self.maxx, 
                          *[ x for x in self.model.optima() if inrange(x) ]])
            y = self.model.predict(self.transform_X(x.reshape(-1,1)))
            y = y.reshape(-1)
            if self.direction == 1:
                i = np.argmax(y)
                self._least_optimal = x[i], y[i]
            else:
                i = np.argmin(y)
                self._least_optimal = x[i], y[i]
            return self._least_optimal
        
    def opt(self):
        a = 3 * self.model.betas[3]
        b = 2 * self.model.betas[2]
        c = self.model.betas[1]
        D = b * b - 4 * a * c
        if D == 0:
            if c < 0 and self.direction == 1:
                return [ -b / (2 * a) ]
            if c > 0 and self.direction == 0:
                return [ -b / (2 * a) ]
        else:
            x = [ (-b - math.sqrt(D))/ (2 * a), (-b + math.sqrt(D))/ (2 * a) ]
            y = self.model.predict(self.transform_X(x))
            if self.direction == 1:
                if y[0] < y[1]:
                    return x[0]
                return x[1]
            else:
                if y[1] < y[0]:
                    return x[1]
                return x[0]

    @property
    def r2(self):
        return r2_score(self.valid_y, self.pred_y)

class OptimumRANSAC(Optimum):
    def __init__(self, study, parameter, target, fold=None, degree=2, min_samples=None,
                residual_threshold=None):
        super().__init__(study, parameter, target, fold=fold, degree=degree)
        self.min_samples = min_samples
        self.residual_threshold = residual_threshold
        
    @property
    def model(self):
        try:
            return self._model
        except:
            self._model = RANSACRegressor(min_samples=self.min_samples, residual_threshold=self.residual_threshold)
            self._model.fit(self.train_X, self.train_y)
            return self._model
        
class OptimumHuber(Optimum):
    def __init__(self, study, parameter, target, fold=None, degree=2, epsilon=1.35,
                alpha=1e-4, tol=1e-5):
        super().__init__(study, parameter, target, fold=fold, degree=degree)
        self.epsilon = epsilon
        self.alpha = alpha
        self.tol = tol
        
    @property
    def model(self):
        try:
            return self._model
        except:
            self._model = HuberRegressor(epsilon=self.epsilon, alpha=self.alpha, tol=self.tol)
            self._model.fit(self.train_X, self.train_y)
            return self._model
        
class OptimumTheilSen(Optimum):
    def __init__(self, study, parameter, target, fold=None, degree=2):
        super().__init__(study, parameter, target, fold=fold, degree=degree)
        
    @property
    def model(self):
        try:
            return self._model
        except:
            self._model = TheilSenRegressor()
            self._model.fit(self.train_X, self.train_y)
            return self._model

class OptimumJ(Optimum):
    def __init__(self, study, parameter, target, fold=None, degree=2, factor=1.0, window=5):
        super().__init__(study, parameter, target, fold=fold, degree=degree)
        self.factor = factor
        self.window = window
        
    @property
    def model(self):
        try:
            return self._model
        except:
            weights = None
            for i in range(len(self.df)):
                mini = max(0, i - window//2)
                maxi = min(len(self.df), i + 1 + window // 2)
                mean = np.mean(self.df.iloc[mini:maxi].targetvalue)
                std = np.std(self.df.iloc[mini:maxi].targetvalue)
                y = np.clip(mean - self.df.iloc[i].targetvalue, None, 0) / std
                weights = st.norm.cdf(y)
            self._model = LinearRegression()
            self._model.fit(self.train_X, self.train_y, sample_weight=weights)
            return self._model

        
