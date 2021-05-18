import torch
import numpy as np
from sklearn.metrics import confusion_matrix, r2_score, f1_score
from .helper import *

class training_module():
    def __init__(self, history):
        self.history = history
        self.trainer = history.trainer
        
    def require_module(self, module):
        self.history.require_module(module)
    
    def requirements(self):
        return False
        
class store_contingencies(training_module):
    "stores a 2x2 contingency table"
    def requirements(self):
        assert self.trainer.out_features == 1

    def after_epoch(self, epoch, y, y_pred, loss):
        confusion_vector = torch.round(y_pred) / torch.round(y)
        epoch['tp'] = torch.sum(confusion_vector == 1).item()
        epoch['fp'] = torch.sum(confusion_vector == float('inf')).item()
        epoch['tn'] = torch.sum(torch.isnan(confusion_vector)).item()
        epoch['fn'] = torch.sum(confusion_vector == 0).item()

class store_confusion(training_module):
    "stores the entire confusion matrix, 1hot encoding must be decoded"
    def requirements(self):
        assert self.trainer.out_features > 1

    def after_epoch(self, epoch, y, y_pred, loss):
        classes = self.trainer.out_features
        yt = y.data.cpu().numpy()
        if len(y_pred.shape) > 1: # check for 1-hot encoding
            y_pred = y_pred.max(1)[1]
        yc = y_pred.data.cpu().numpy()
        cm = confusion_matrix(yt, yc, labels=range(classes))
        epoch['cm'] = cm
