import torch
from torch import nn
import torchvision.models as models
from sklearn.metrics import *
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from .data.dframe import DFrame
from .data.databunch import Databunch
from .data.textcollection import TextCollection
from .data.datasets import read_torchtext, wine_quality, telco_churn, movielens_ratings, movielens_movies, movielens_users, dam_outflow, boston_housing_prices, iris, bank_marketing, auto_mpg, big_mart_sales, advertising_sales, titanic, indian_liver, air_passengers, ames_housing, diamonds, california, occupancy, ag_news, ames_housing, boston_housing_prices, heart_disease, hotel_booking, hotel_test, hotel_test_score, bbc_news, speeddate
from .data.imagedframe import ImageDFrame, ImageDatabunch
from .data.imagedatasets import mnist, mnist3, crawl_images, filter_images, image_folder, create_path, cifar, fashionmnist
from .data.kagglereader import Kaggle, create_kaggle_authentication
from .train.trainer import Trainer, Trainer as trainer
from .model.perceptron import Perceptron
from .model.convnet import ConvNet
from .model.transfer import Transfer, models

#from .data import *
#from .evaluate import *
#from .model import *
#from .train import *
#from .gpu import *
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

