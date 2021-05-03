from .ptdataframe import PTDataFrame, read_csv, read_excel, read_pd_csv, PTGroupedDataFrame
from .databunch import Databunch
from .datasets import wine_quality, telco_churn, movie_ratings, dam_outflow, boston_housing_prices, iris, bank_marketing, auto_mpg, big_mart_sales, advertising_channels, titanic_survivors, indian_liver, flight_passengers, ames_housing, datasets, diamonds, california, movie_titles, nyse50, occupancy
from .evaluate import Evaluator
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

