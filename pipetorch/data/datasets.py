from .ptdataframe import PTDataFrame
from .textcollection import TextCollection
from torchtext.datasets import AG_NEWS
from pathlib import Path
from getpass import getuser
import pandas as pd
import numpy as np
import pickle
import path
import os
from sklearn.datasets import load_boston, load_iris
import requests
from io import BytesIO
from zipfile import ZipFile
from functools import partial
from google_images_download import google_images_download
from tqdm.notebook import tqdm
import ipywidgets as widgets
import io
from PIL import Image, ImageStat

def _gis_args(keywords, output_directory=None, 
                 image_directory=None, limit=200, format='jpg', color_type='full-color', 
                 size='medium', type='photo', delay=0, **kwargs):
    if output_directory is None:
        output_directory = str(create_path(f'/tmp/{getuser()}/images'))
    if image_directory is None:
        image_directory = '_'.join(keywords.split())
    arguments = {"keywords":keywords, 
                 "limit":limit, "format":format, "color_type":color_type, "size":size, "type":type, 
                 "delay":delay, "image_directory":image_directory, 
                 "output_directory":output_directory, "chromedriver":"/usr/bin/chromedriver" }
    arguments.update(kwargs)
    return arguments

def crawl_images(keywords, output_directory=None, 
                 image_directory=None, limit=200, format='jpg', color_type='full-color', 
                 size='medium', type='photo', delay=0, **kwargs):
    """
    Downloads images through Google Image Search, 
    see https://google-images-download.readthedocs.io/en/latest/arguments.html 
    for info on the arguments. When no output_directory is given, the downloaded images
    are stored in /tmp/<username>/images/<query>.
    """
    kwargs = _gis_args(keywords, output_directory=output_directory, image_directory=image_directory, 
             limit=limit, format=format, color_type=color_type, size=size, type=type, delay=delay, 
             **kwargs)
    response = google_images_download.googleimagesdownload()   #class instantiation
    paths = response.download(kwargs)   #passing the arguments to the function
    
def filter_images(keywords, folder=None, columns=4, height=200, width=200):
    """
    Removes duplicate images and shows the remaining images so that the user can manually select
    images to remove from the folder by pressing the DELETE button below.
    """
    def on_click(button):
        for r in rows:
            if type(r) is widgets.HBox:
                for c in r.children:
                    checkbox = c.children[1]
                    if checkbox.value:
                        print(checkbox.description_tooltip)
                        os.remove(checkbox.description_tooltip)

    if folder is None:
        folder = Path(f'/tmp/{getuser()}/images')
    keywords = '_'.join(keywords.split())
    imagefiles = [f for f in folder.glob(keywords + '/*')]
    rows = []
    cols = []
    bymean = {}
    for i, imgfile in enumerate(tqdm(imagefiles)):
        row = i // columns
        col = i % columns
        img = Image.open(imgfile)
        m = hash(tuple(ImageStat.Stat(img).mean))
        buff = io.BytesIO()   
        img.save(buff, format='JPEG')
        if m in bymean:
            os.remove(imgfile)
        else:
            bymean[m] = imgfile

        image = widgets.Image( value=buff.getvalue(), width=width, height=height )
        button = widgets.Checkbox( description='Delete', description_tooltip = str(imgfile) )
        box = widgets.VBox([image, button])
        cols.append(box)
        if len(cols) == columns:
            rows.append(widgets.HBox(cols))
            cols = []
                 
    if len(cols) > 0:
        rows.append(widgets.HBox(cols))
    button = widgets.Button( description='Delete' )
    button.on_click(on_click)
    rows.append(button)
    return widgets.VBox(rows)        

def create_path(p, mode=0o777):
    path = Path(p)
    os.makedirs(path, mode, exist_ok=True)
    return path

def path_user():
    return Path.home() / '.pipetorchuser'

def path_shared():
    return Path.home() / '.pipetorch'
    
def get_filename(url):
    fragment_removed = url.split("#")[0]  # keep to left of first #
    query_string_removed = fragment_removed.split("?")[0]
    scheme_removed = query_string_removed.split("://")[-1].split(":")[-1]
    if scheme_removed.find("/") == -1:
        filename = scheme_removed
    else:
        filename = os.path.basename(scheme_removed)
    if '.' in filename:
        filename = filename.rsplit( ".", 1 )[ 0 ] + '.csv'
    return filename
    
def read_excel(path, filename=None, alternativesource=None, sep=None, delimiter=None, **kwargs):
    if filename is None:
        filename = get_filename(path)
    if (path_user() / filename).is_file():
        return PTDataFrame.read_csv(path_user() / filename, **kwargs)
    if (path_shared() / filename).is_file():
        return PTDataFrame.read_csv(path_shared() / filename, **kwargs)
    if alternativesource:
        df = pd.read_excel(alternativesource())
    else:
        print('Downloading new file ' + path)
        df = pd.read_excel(path, **kwargs)
        df.columns = df.columns.str.replace(' ', '') 
    (path_user()).mkdir(exist_ok=True)
    df.to_csv(path_user() / filename, index=False)
    return PTDataFrame(df)

def read_pd_csv(path, filename=None, alternativesource=None, sep=None, delimiter=None, **kwargs):
    if sep:
        kwargs['sep'] = sep
    elif delimiter:
        kwargs['delimiter'] = delimiter
    if filename is None:
        filename = get_filename(path)
    if (path_user() / filename).is_file():
        #print(str(Path.home() / '.pipetorchuser' / filename))
        return pd.read_csv(path_user() / filename, **kwargs)
    if (path_shared() / filename).is_file():
        #print(str(Path.home() / '.pipetorch' / filename))
        return pd.read_csv(path_shared() / filename, **kwargs)
    if alternativesource:
        df = alternativesource()
    else:
        print('Downloading new file ' + path)
        df = pd.read_csv(path, **kwargs)
    (path_user()).mkdir(exist_ok=True)
    df.to_csv(path_user() / filename, index=False)
    return df

def read_csv(path, filename=None, alternativesource=None, sep=None, delimiter=None, **kwargs):
    return PTDataFrame(read_pd_csv(path, filename=filename, alternativesource=alternativesource, sep=sep, delimiter=delimiter, **kwargs))

def read_torchtext(torchtext_function):
    try:
        return torchtext_function(root=path_shared() / torchtext_function.__name__)
    except:
        return torchtext_function(root=path_user() / torchtext_function.__name__)

def wine_quality():
    return read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv', delimiter=';')

def telco_churn():
    return read_csv('https://github.com/pmservice/wml-sample-models/raw/master/spark/customer-satisfaction-prediction/data/WA_Fn%20UseC_%20Telco%20Customer%20Churn.csv', filename='telco_churn.csv')

def movie_ratings():
    def read():
        COLS = ['user_id', 'movie_id', 'rating', 'timestamp']
        return pd.read_csv("https://raw.githubusercontent.com/ChicagoBoothML/DATA___MovieLens___1M/master/ratings.dat",sep='::', engine='python', names=COLS)
    return read_csv("movielens1M.csv", alternativesource=read)

def movie_titles():
    def read():
        COLS = ['movie_id', 'title', 'genre']
        return pd.read_csv("https://raw.githubusercontent.com/ChicagoBoothML/DATA___MovieLens___1M/master/movies.dat",sep='::', engine='python', encoding="iso-8859-1", names=COLS)
    return read_csv("movies1M.csv", alternativesource=read)
    
def dam_outflow():
    try:
        with open("/data/datasets/dam_water_data.pickle", "rb") as myfile:
            X_train, X_val, X_test, X_all, y_train, y_val, y_test, y_all = pickle.load(myfile)
            train_indices = [ i for i, v in enumerate(X_all) if (X_train == v).any() ]
            X_all = X_all.astype(np.float32)
            y_all = y_all.astype(np.float32)
            df = PTDataFrame(np.concatenate([X_all, y_all.reshape(-1, 1)], axis=1), columns=['waterlevel', 'outflow'])
        return df
    except:
        print('This dataset is not online, but was taken from Andrew Ng\'s Coursera course')

def boston_housing_prices():
    """
    Load the Boston Housing Prices dataset and return it as a Pandas Dataframe
    """
    boston = load_boston()
    df = pd.DataFrame(boston['data'] )
    df.columns = boston['feature_names']
    df['PRICE'] = boston['target']
    return PTDataFrame(df)

def iris():
    iris=load_iris()
    df = pd.DataFrame(data=np.c_[iris['data'], iris['target']],
                      columns= iris['feature_names'] + ['target'])
    return PTDataFrame(df)

def bank_marketing():
    return read_csv("https://github.com/llhthinker/MachineLearningLab/raw/master/UCI%20Bank%20Marketing%20Data%20Set/data/bank-additional/bank-additional-full.csv", filename='bank_marketing.csv', sep=';')

def auto_mpg():
    return read_csv('https://raw.githubusercontent.com/joanby/python-ml-course/master/datasets/auto/auto-mpg.csv')

def big_mart_sales():
    return read_csv('https://raw.githubusercontent.com/akki8087/Big-Mart-Sales/master/Train.csv', filename='big_mart_sales.csv')

def advertising_channels():
    return read_csv('https://raw.githubusercontent.com/nguyen-toan/ISLR/master/dataset/Advertising.csv').iloc[:,1:]

def titanic_survivors():
    return read_csv('https://web.stanford.edu/class/archive/cs/cs109/cs109.1166/stuff/titanic.csv')

def diamonds():
    return read_csv('https://raw.githubusercontent.com/SiphuLangeni/Diamond-Price-Prediction/master/Diamonds.csv')

def indian_liver():
    return read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/00225/Indian%20Liver%20Patient%20Dataset%20(ILPD).csv')
                   #names=["Age", "Gender", "Total Bilirubin", "Direct Bilirubin", "Alkphos Alkaline Phosphotase", "Sgpt Alamine Aminotransferase", "Sgot Aspartate Aminotransferase", "Total Protiens", "Albumin", "Albumin-Globulin Ratio", "Disease"])

def ames_housing():
    return read_excel('http://www.amstat.org/publications/jse/v19n3/decock/AmesHousing.xls')
    
def flight_passengers():
    import seaborn as sns
    df = sns.load_dataset('flights')
    #df['month'] = df.month.map({'Jan':0, 'Feb':1, 'Mar':2, 'Apr':3, 'May':4, 'Jun':5, 'Jul':6, 'Aug':7, 'Sep':8, 'Oct':9, 'Nov':10, 'Dec':11}).astype(np.float32)
    return PTDataFrame(df)

def rossmann():
    def read():
        df = pd.read_csv('https://raw.githubusercontent.com/sarthaksoni25/Rossmann-Store-Sales-Prediction/master/dataset/train.csv')
        return df['Id', 'Store', 'Date', 'DayOfWeek', 'Customers', 'Open', 'Promo', 'StateHoliday', 'SchoolHoliday', 'Sales']
    return read_csv('rossmann.csv', alternativesource=read)

def california():
    return read_csv('https://raw.githubusercontent.com/subhadipml/California-Housing-Price-Prediction/master/housing.csv', filename='california')
                         
def flights():
    df = sns.load_dataset('flights')
    df['month'] = df.month.map({'Jan':0, 'Feb':1, 'Mar':2, 'Apr':3, 'May':4, 'Jun':5, 'Jul':6, 'Aug':7, 'Sep':8, 'Oct':9, 'Nov':10, 'Dec':11}).astype(np.float32)
    return PTDataFrame(df)

def nyse50(**kwargs):
    df = pd.read_csv('/data/datasets/nyse-top50.csv', **kwargs)
    return PTDataFrame(df)
    
def occupancy():
    def read(i):
        url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/00357/occupancy_data.zip'
        content = requests.get(url)

        f = ZipFile(BytesIO(content.content))
        with f.open(f.namelist()[i], 'r') as g:     
            return pd.read_csv(g)
    train = read_pd_csv('occupancy_train.csv', alternativesource=partial(read, 2))
    valid = read_pd_csv('occupancy_valid.csv', alternativesource=partial(read, 0))
    test = read_pd_csv('occupancy_test.csv', alternativesource=partial(read, 1))
    return PTDataFrame.from_dfs(train, valid=valid, test=test)

def ag_news(valid_perc=0.05, language='basic_english', min_freq=1, collate='pad', shuffle=True):
    train_iter, test_iter = read_torchtext(AG_NEWS)
    tc = TextCollection.from_iter(train_iter, None, test_iter, min_freq=min_freq, shuffle=shuffle).split(valid_perc).collate(collate)
    return tc

_ptdatasetslist = [('Indian Liver Disease', 'pt.indian_liver()', 'https://archive.ics.uci.edu/ml/datasets/ILPD+(Indian+Liver+Patient+Dataset)'),
            ('Historial flight passengers', 'pt.flight_passengers()', 'From Seaborn library'),
            ('Advertising channels', 'pt.advertising_channels()', 'https://www.kaggle.com/ashydv/advertising-dataset'),
            ('Titanic survival', 'pt.titanic()', 'https://www.kaggle.com/c/titanic'),
            ('Big Mart Sales', 'pt.big_mart_sales()', 'https://medium.com/total-data-science/big-mart-sales-data-science-projects-98919293c1b3'),
            ('Auto MPG', 'pt.auto_mpg()', 'https://archive.ics.uci.edu/ml/datasets/auto+mpg'),
            ('Bank Marketing', 'pt.bank_marketing()', 'https://archive.ics.uci.edu/ml/datasets/bank+marketing'),
            ('Iris', 'pt.iris()', 'https://archive.ics.uci.edu/ml/datasets/iris'),
            ('Boston Housing Prices', 'pt.boston_housing_prices()', 'https://www.cs.toronto.edu/~delve/data/boston/bostonDetail.html'),
            ('Movie Ratings', 'pt.movie_ratings()', 'https://grouplens.org/datasets/movielens/1m/'),
            ('Wine Quality', 'pt.wine_quality()', 'https://archive.ics.uci.edu/ml/datasets/wine+quality'),
            ('Telco Churn', 'pt.telco_churn()', 'https://www.kaggle.com/blastchar/telco-customer-churn'),
            ('Dam water outflow', 'pt.dam_outflow()', 'From Andrew Ng\'s Coursera Course'),
            ('Kaggle House Prices Competition', 'pt.house_prices()', 'http://www.amstat.org/publications/jse/v19n3/decock/AmesHousing.xls'),
            ('Rossmann Store Sales', 'pt.rossmann()', 'https://www.kaggle.com/c/rossmann-store-sales'),
            ('California Housing', 'pt.california()', 'https://www.kaggle.com/camnugent/california-housing-prices'),
            ('Flights', 'pt.flights()', 'Example Dataset from the Seaborn library with the number of passengers per month'),
            ('NYSE', 'pt.nyse50()', 'Crawl of the 2020 quotes of the 50 stocks with the highest turnover on the New York Stock Exchange'),
            ('Room occupancy', 'pt.occupancy()', 'https://archive.ics.uci.edu/ml/datasets/Occupancy+Detection+'),
            ('AG News', 'pt.ag_news()', 'https://pytorch.org/text/stable/datasets.html#ag-news')
           ]
datasets = pd.DataFrame(_ptdatasetslist, columns=['dataset', 'method', 'url'])