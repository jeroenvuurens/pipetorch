from .data import *
from .evaluate import *
from .model import *
from .train import *
#from .gpu import *
import matplotlib.pyplot as plt
from .version import __version__
import warnings

class catch_warnings:
    def __enter__(self):
        warnings.filterwarnings("error")
        return self

    def __exit__(self, exc_type, exc_value, exc_traceback):
        warnings.filterwarnings("default")

def list_all(s):
    try:
        return s.__all__
    except:
        return [ o for o in dir(s) if not o.startswith('_') ]

#subpackages = [ jtorch.train, jtorch.train_modules, jtorch.train_diagnostics, jtorch.train_metrics, jtorch.jcollections ]

#subpackages = [ trainer ]

#__all__ = [ f for s in subpackages for f in list_all(s) ]

