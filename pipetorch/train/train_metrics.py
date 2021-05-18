from .train_modules import *
from sklearn.metrics import *
import math

def name_metrics(metrics):
    for i, m in enumerate(metrics):
        try:
            m, m.__name__ = m.value, m.__name__
        except:
            m.__name__ = f'm_{i}'
        yield m

class train_metrics():
    def __init__(self, history=None):
        self.history = history
        
    @property
    def __name__(self):
        return self.__name()

    def __name(self):
        return self.__class__.__name__
  
    @classmethod
    def value(cls, epoch):
        try:
            return epoch[cls.__name__]
        except:
            return cls.value2(epoch)
        
class acc(train_metrics):
    @staticmethod
    def value2(epoch):
        return accuracy_score(epoch.y, epoch.y_pred.reshape(epoch.y.shape).round())

class acc_mc(train_metrics):
    @staticmethod
    def value2(epoch):
        return accuracy_score(epoch.y, np.argmax(epoch.y_pred, axis=1))

class recall(train_metrics):
    @staticmethod
    def value2(epoch):
        return recall_score(epoch.y, epoch.y_pred.reshape(epoch.y.shape).round())

class recall4(train_metrics):
    @staticmethod
    def value2(epoch):
        return recall_score(epoch.y > 1, epoch.y_pred.reshape(epoch.y.shape).round())

class precision(train_metrics):
    @staticmethod
    def value2(epoch):
        return precision_score(epoch.y, epoch.y_pred.reshape(epoch.y.shape).round())

class precision4(train_metrics):
    @staticmethod
    def value2(epoch):
        return precision_score(epoch.y > 1, epoch.y_pred.reshape(epoch.y.shape).round())

class f1_4(train_metrics):
    @staticmethod
    def value2(epoch):
        return f1_score(epoch.y > 1, epoch.y_pred.reshape(epoch.y.shape).round())

class fp_4(train_metrics):
    @staticmethod
    def value2(epoch):
        return fbeta_score(epoch.y > 1, epoch.y_pred.reshape(epoch.y.shape).round(), beta=0.2)

class f1(train_metrics):
    @staticmethod
    def value2(epoch):
        return f1_score(epoch.y, epoch.y_pred.reshape(epoch.y.shape).round())

class fp(train_metrics):
    @staticmethod
    def value2(epoch):
        return fbeta_score(epoch.y, epoch.y_pred.reshape(epoch.y.shape).round(), beta=0.2)

class mse(train_metrics):
    @staticmethod
    def value2(epoch):
        return mean_squared_error(epoch.y, epoch.y_pred.reshape(epoch.y.shape))

class rmse(train_metrics):
    @staticmethod
    def value2(epoch):
        return math.sqrt(mean_squared_error(epoch.y, epoch.y_pred.reshape(epoch.y.shape)))

class r2(train_metrics):
    @staticmethod
    def value2(epoch):
        return r2_score(epoch.y, epoch.y_pred.reshape(epoch.y.shape))

class loss(train_metrics):
    @staticmethod
    def value2(epoch):
        return epoch['loss']    
    

