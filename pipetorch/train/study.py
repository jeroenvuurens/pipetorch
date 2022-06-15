import optuna
from sklearn.linear_model import SGDRegressor
from sklearn.preprocessing import PolynomialFeatures, MinMaxScaler
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import inspect
from collections import defaultdict
from functools import partial

class Study(optuna.study.Study):
    """
    Extension to an optuna Study. This extension caches the target functions and plot_hyperparameters
    provides a good side-by-side overvew of the hyperparameters over the targets.
    
    For more information, check out Optuna Study.
    """
    
    def __init__(self, study, *target, trainer=None, grid=None):
        """
        Call create_study to instantiate a study
        """
        super().__init__(study.study_name, study._storage, study.sampler, study.pruner)
        assert len(target) > 0, 'You need to define at least one target'
        for t in target:
            assert type(t) == str, 'Only str names for targets are currently supported'
        self.target = target
        self.trainer = trainer
        self.grid = grid
        
    @classmethod
    def create_study(cls, *target, trainer=None, storage=None, sampler=None, pruner=None, 
                     study_name=None, direction=None, directions=None, load_if_exists=False, grid=None):
        """
        Uses optuna.create_study to create a Study. This extension registers the target metrics for inspection.
        
        Arguments:
            *target: 'loss' or str or callable or Trainer
                When called with no targets, this is set to 'loss'
                When called with a trainer, this is set to 'loss' + all metrics that are registered by the trainer
                Otherwise call with a sequence of callables or strings in the same order they are registered by
                the Trainer that is used, e.g. a `Trainer(metrics='f1_score')` will have the `optimum` function return
                `(loss, f1_score)`, therefore, register the study with `Study.create_study('loss', f1_score)`.
                When direction is omitted, loss is set to minimize and all other directions to maximize.
            other arguments: check optuna
        """
        if grid is not None:
            assert type(grid) == dict, 'You have to pass a dict to grid'
            sampler = optuna.samplers.GridSampler(grid)
        if len(target) == 0:
            target = ['loss']
        if len(target) == 1:
            from .trainer import Trainer
            if type(target[0]) == Trainer:
                trainer=target[0]
                target = ['loss'] + [ m.__name__ for m in target[0].metrics ]
        if direction is None and directions is None:
            if len(target) > 1:
                directions = [ 'minimize' if t == 'loss' else 'maximize' for t in target ]
            else:
                direction = 'minimize' if target[0] == 'loss' else 'maximize'
        study = optuna.create_study(storage=storage, sampler=sampler, pruner=pruner,
                                    study_name=study_name, direction=direction, directions=directions, 
                                    load_if_exists=load_if_exists)
        return cls(study, *target, trainer=trainer, grid=grid)
    
    def ask(self, fixed_distributions=None):
        fixed_distributions = fixed_distributions or {}

        # Sync storage once every trial.
        self._storage.read_trials_from_remote_storage(self._study_id)

        trial_id = self._pop_waiting_trial_id()
        if trial_id is None:
            trial_id = self._storage.create_new_trial(self._study_id)
        trial = Trial(self, trial_id)

        for name, param in fixed_distributions.items():
            trial._suggest(name, param)

        return trial
    
    def optimize(self, func, n_trials=None, timeout=None, catch=(), callbacks=None, 
                 gc_after_trial=False, show_progress_bar=False):
        """
        See Optuna's optimize, this extensions adds passing the trainer to the trial function.
        """
        
        args = len(inspect.getargspec(func)[0])
        if args == 2:
            assert self.trainer is not None, 'You can only pass a func with two arguments when trainer is set'
            func = partial(func, self.trainer)
        super().optimize(func, n_trials=n_trials, timeout=timeout, catch=catch, callbacks=callbacks,
                      gc_after_trial=gc_after_trial, show_progress_bar=show_progress_bar)        
    
    def __repr__(self):
        return repr(self.validate())
    
    def filter_targets(self, results):
        return [ results[t] for t in self.target ]
    
    def parameters(self):
        return self.trials[0].params.keys()

    def results(self):
        table = []
        for t in self.trials:
            for param, paramv in t.params.items():
                for target, value in zip(self.target, t.values):
                    table.append((t.number, param, paramv, target, value))
        return pd.DataFrame(table, columns=['trial', 'parameter', 'parametersetting', 'target', 'targetvalue'])

    
    def distribution(self, param):
        dist = self.trials[0].distributions[param]
        return dist
    
    def is_log_distribution(self, param):
        return self.distribution(param).__class__.__name__.startswith('Log')
    
    def plot_hyperparameters(self, figsize=None, logscale=['loss']):
        """
        Plots the sensitivity of each hyperparameter over each recorded metric.
        
        Arguments:
            figsize: (width, height) None
                controls the size of the figure displayed
            logscale: ['loss']
                list of metrics whose y-axis is shown as a log scale. By default this is done for the loss
                because the learning rate is often sampled from a log distribution and this makes it easier
                to estimate the optimum.
        """
        results = self.results()
        parameters = self.parameters()
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
                subset = results[(results.parameter == param) & (results.target == target)]
                self._subplot(axs[targeti, parami], subset)
                if targeti == 0:
                    axs[targeti, parami].set_title(param)
                    if self.is_log_distribution(param):
                        axs[targeti, parami].set_xscale('log')
                if target in logscale:
                    axs[targeti, parami].set_yscale('log')
                if parami == 0:
                    axs[targeti, parami].set_ylabel(target)
    
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
    
    def _subplot(self, ax, subset):
        x = subset.parametersetting.astype(np.float64)
        y = subset.targetvalue.astype(np.float64)
        z = subset.trial
        ax.scatter(x, y, c=z, cmap='plasma')
        
    def plot(self):
        optuna.visualization.plot_slice(self, params=["hidden"],
                                  target_name="F1 Score")
    
class Trial(optuna.trial.Trial):
    def suggest_lr(self, name, low, high):
        sequence = [ l * 10 for l in range ]
        
    def suggest_categorical(self, name, choices=None):
        try:
            choices = choices or self.study.grid[name]
        except: pass
        return super().suggest_categorical(name, choices)
