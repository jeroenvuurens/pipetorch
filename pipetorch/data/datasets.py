from .dframe import DFrame
from .textcollection import TextCollection
from pathlib import Path
from getpass import getuser
import pandas as pd
import numpy as np
import pickle
import shutil
import os
from io import StringIO
import pkgutil
from sklearn.datasets import load_boston, load_iris
from sklearn.metrics import f1_score
from io import BytesIO
from functools import partial
from zipfile import ZipFile

def path_user(dataset=None):
    if dataset is not None:
        return Path.home() / '.pipetorchuser' / dataset.split('/')[-1]
    return Path.home() / '.pipetorchuser'

def path_shared(dataset=None):
    if dataset is not None:
        return Path.home() / '.pipetorch' / dataset.split('/')[-1]    
    return Path.home() / '.pipetorch'

def dataset_path(dataset):
    try:
        p = path_shared(dataset)
        if p.exists():
            return p
    except: pass
    return path_user(dataset)

def get_stored_path(filename, path=None):
    if path is not None:
        storedpath = path / filename
    else:
        storedpath = (path_user() / filename)
        if not storedpath.exists():
            storedpath = (path_shared() / filename)
        if not storedpath.exists():
            storedpath = (path_user() / filename)
    return storedpath

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

def to_csv(df, filename, **kwargs):
    kwargs = { key:value for key, value in kwargs.items() if key in {'sep', 'quoting', 'quotechar', 'lineterminator', 'decimal', 'line_terminator', 'doublequote', 'escapechar'}}
    kwargs['index'] = False
    if 'sep' in kwargs and len(kwargs['sep']) > 1:
        sep = kwargs['sep']
        kwargs['sep'] = '¤'
        csv = df.to_csv(**kwargs).replace('¤', sep)
        with open(filename, 'w') as fout:
            fout.write(csv)
    else:
        df.to_csv(filename, **kwargs)   

def read_pd_excel(path, filename=None, save=True, **kwargs):
    if filename is None:
        filename = get_filename(path)
    if (path_user() / filename).is_file():
        return pd.read_excel(path_user() / filename, **kwargs)
    if (path_shared() / filename).is_file():
        return pd.read_excel(path_shared() / filename, **kwargs)
    #print('Downloading new file ' + path)
    df = pd.read_excel(path, **kwargs)
    df.columns = df.columns.str.replace(' ', '') 
    return df

def read_excel(path, filename=None, **kwargs):
    return DFrame(read_pd_excel(path, filename=filename, **kwargs))

def create_kaggle_authentication(username, key):
    """
    Writes the authentication by kaggle to a file, so that PipeTorch can use the kaggle api to 
    download datasets.
    
    Arguments:
        username: str
            your username at kaggle (you do need to register)
        key: str
            the token that you have generated on your kaggle account (can be read in the .json file)
    """
    if not os.path.exists(Path.home() / '.kaggle'):
        os.path.mkdir(Path.home() / '.kaggle')
    os.path.chmod(Path.home() / '.kaggle', 0o600)
    kagglejson = Path.home() / '.kaggle' / '.kaggle.json'
    if not os.path.exists(kagglejson):
        with open (kagglejson, "w") as fout:
            fout.write(f'{"username":"{username}", "key":"{key}"}')
            print(f'kaggle authorization written to {kagglejson}')
        os.path.chmod('~/.kaggle/kaggle.json', 0o600)
    else:
        print(f'kaggle authorization already exists, check {kagglejson}')

def kaggle_download(dataset, shared=False):
    try:
        from kaggle.api.kaggle_api_extended import KaggleApi
        api = KaggleApi()
        api.authenticate()
    except:
        print('''
        Error authenticating kaggle: you need to (1) register at kaggle (2) generate a token/key and 
        (3) put your user credentials in ~/.kaggle/kaggle.json. You can perform (3) using 
        pipetorch.data.create_kaggle_authentication(username, key)
        ''')
        return
    path = path_shared(dataset) if shared else dataset_path(dataset)
    print(f'Downloading {dataset} from kaggle to {path}')
    api.dataset_download_files(dataset, path=path, unzip=True)

def kaggle_download_competition(dataset, shared=False):
    try:
        from kaggle.api.kaggle_api_extended import KaggleApi
        api = KaggleApi()
        api.authenticate()
    except:
        print('''
        Error authenticating kaggle: you need to (1) register at kaggle (2) generate a token/key and 
        (3) put your user credentials in ~/.kaggle/kaggle.json. You can perform (3) using 
        pipetorch.data.create_kaggle_authentication(username, key)
        ''')
        return
    path = path_shared(dataset) if shared else dataset_path(dataset)
    print(f'Downloading {dataset} from kaggle to {path}')
    api.competition_download_files(dataset, path=path)
    for zfile in list(path.glob('*.zip')):
        zip = ZipFile(zfile)
        zip.extractall(path=path)
        zfile.unlink()

def read_pd_from_kaggle(dataset, filename=None, shared=False, force=False, **kwargs):
    if force:
        try:
            path = path_user(dataset)
            shutil.rmtree(path)
        except: pass
    path = dataset_path(dataset)
    if force or not path.exists():
        kaggle_download(dataset, shared=shared)
    path = dataset_path(dataset)
    assert path.exists(), f'Problem downloading Kaggle dataset {dataset}'
    if filename is None:
        filename = '**/*'
    files = list(path.glob(filename))
    assert len(files) == 1, f'There are multiple files that match {files}, set filename to select a file'
    return pd.read_csv(path / files[0], **kwargs)

def read_pd_from_kaggle_competition(dataset, filename=None, shared=False, force=False, **kwargs):
    if force:
        try:
            path = path_user(dataset)
            shutil.rmtree(path)
        except: pass
    path = dataset_path(dataset)
    if force or not path.exists():
        kaggle_download_competition(dataset, shared=shared)
    path = dataset_path(dataset)
    assert path.exists(), f'Problem downloading Kaggle dataset {dataset}'
    if filename is None:
        filename = '**/*'
    files = list(path.glob(filename))
    assert len(files) == 1, f'There are multiple files that match {files}, set filename to select a file'
    return pd.read_csv(path / files[0], **kwargs)

def read_from_kaggle(dataset, train=None, test=None, shared=False, force=False, **kwargs):
    """
    Reads a DFrame from a Kaggle dataset. The downloaded dataset is automatically stored so that the next time
    it is read from file rather than downloaded. See `read_csv`. The dataset is stored by default in a folder
    with the dataset name in `~/.pipetorchuser`.
    
    If the dataset is not cached, this functions requires a valid .kaggle/kaggle.json file, that you can 
    create manually or with the function `create_kaggle_authentication()`.

    Note: there is a difference between a Kaggle dataset and a Kaggle competition. For the latter, 
    you have to use `read_from_kaggle_competition`.
    
    Example:
        read_from_kaggle('uciml/autompg-dataset')
            to read/download `https://www.kaggle.com/datasets/uciml/autompg-dataset`
        read_from_kaggle('robmarkcole/occupancy-detection-data-set-uci', 'datatraining.txt', 'datatest.txt')
            to combine a train and test set in a single DFrame
    
    Arguments:
        dataset: str
            the username/dataset part of the kaggle url, e.g. uciml/autompg-dataset for 
            
        train: str (None)
            the filename that is used as the train set, e.g. 'train.csv'
        test: str (None)
            the filename that is used as the test set, e.g. 'test.csv'
        shared: bool (False)
            save the dataset in ~/.pipetorch instead of ~/.pipetorchuser, allowing to share downloaded
            files between users.
        force: bool (False)
            when True, the dataset is always downloaded
        **kwargs:
            additional parameters passed to pd.read_csv. For example, when a multichar delimiter is used
            you will have to set engine='python'.
            
    Returns: DFrame
    """
    train = read_pd_from_kaggle(dataset, filename=train, shared=shared, force=force, **kwargs)
    if test is not None:
        test = read_pd_from_kaggle(dataset, filename=test, **kwargs)
        return DFrame.from_train_test(train, test)
    return DFrame(train)

def read_from_kaggle_competition(dataset, train=None, test=None, shared=False, force=False, **kwargs):
    train = read_pd_from_kaggle_competition(dataset, filename=train, shared=shared, force=force, **kwargs)
    if test is not None:
        test = read_pd_from_kaggle_competition(dataset, filename=test, **kwargs)
        return DFrame.from_train_test(train, test)
    return DFrame(train)

def read_pd_csv(url, filename=None, path=None, save=False, **kwargs):
    """
    Reads a .csv file from cache or url. The place to store the file is indicated by path / filename
    and when a delimiter is used, this is also used to save the file so that the original delimiter is kept.
    The file is only downloaded using the url if it does not exsists on the filing system. If the file is
    downloaded and save=True, it is also stored for future use.
    
    Arguments:
        url: str
            the url to download or a full path pointing to a .csv file
        filename: str (None)
            the filename to store the downloaded file under. If None, the filename is extracted from the url.
        path: str (None)
            the path in which the file is stored. If None, it will first check the ~/.pipetorch (for sharing
            dataset between users) and then ~/.pipetorchuser (for user specific caching of datasets).
        save: bool (False)
            whether to save a downloaded .csv
        **kwargs:
            additional parameters passed to pd.read_csv. For example, when a multichar delimiter is used
            you will have to set engine='python'.
            
    Returns: pd.DataFrame
    """
    if filename is None:
        filename = get_filename(url)
    storedpath = get_stored_path(filename, path)        
    if not storedpath.exists():
        storedpath = (path_user() / filename)
        (path_user()).mkdir(exist_ok=True)
        if '://' in url:
            print(f'Downloading {url}')
        df = pd.read_csv(url, **kwargs)
        if save:
            print(f'saving to {storedpath}')
            to_csv(df, filename, **kwargs)
        return df
    else:
        return pd.read_csv(storedpath, **kwargs)

def read_csv(url, filename=None, path=None, save=False, **kwargs):
    """
    Reads a .csv file from cache or url. The place to store the file is indicated by path / filename
    and when a delimiter is used, this is also used to save the file so that the original delimiter is kept.
    The file is only downloaded using the url if it does not exsists on the filing system. If the file is
    downloaded and save=True, it is also stored for future use.
    
    Arguments:
        url: str
            the url to download or a full path pointing to a .csv file
        filename: str (None)
            the filename to store the downloaded file under. If None, the filename is extracted from the url.
        path: str (None)
            the path in which the file is stored. If None, it will first check the ~/.pipetorch (for sharing
            dataset between users) and then ~/.pipetorchuser (for user specific caching of datasets).
        save: bool (False)
            whether to save a downloaded .csv
        **kwargs:
            additional parameters passed to pd.read_csv. For example, when a multichar delimiter is used
            you will have to set engine='python'.
            
    Returns: DFrame
    """
    return DFrame(read_pd_csv(url, filename=filename, path=path, save=save, **kwargs))
    
def read_pd_from_package(package, filename, **kwargs):
    csv = pkgutil.get_data(package, filename).decode()
    return pd.read_csv(StringIO(csv), **kwargs)

def read_from_package(package, filename, **kwargs):
    return DFrame(read_pd_from_package(package, filename))

def read_pd_from_function(filename, function, path=None, save=True, **kwargs):
    """
    First checks if a .csv file is already stored, otherwise, calls the custom function to retrieve a 
    DataFrame. 
    
    The place to store the file is indicated by path / filename.
    The file is only retrieved from the function if it does not exsists on the filing system. 
    If the file is retrieved and save=True, it is also stored for future use.
    
    Arguments:
        filename: str (None)
            the filename to store the downloaded file under.
        function: func
            a function that is called to retrieve the DataFrame if the file does not exist.
        path: str (None)
            the path in which the file is stored. If None, it will first check the ~/.pipetorch (for sharing
            dataset between users) and then ~/.pipetorchuser (for user specific caching of datasets).
        save: bool (True)
            whether to save a downloaded .csv
        **kwargs:
            additional parameters passed to pd.read_csv. For example, when a multichar delimiter is used
            you will have to set engine='python'.
            
    Returns: pd.DataFrame
    """
    storedpath = get_stored_path(filename, path)
    if storedpath.is_file():
        return pd.read_csv(storedpath, **kwargs)
    df = function()
    to_csv(df, storedpath, **kwargs)
    return df
    
def read_from_function(filename, function, path=None, save=True, **kwargs):
    """
    First checks if a .csv file is already stored, otherwise, calls the custom function to retrieve a 
    DataFrame. 
    
    The place to store the file is indicated by path / filename.
    The file is only retrieved from the function if it does not exsists on the filing system. 
    If the file is retrieved and save=True, it is also stored for future use.
    
    Arguments:
        filename: str (None)
            the filename to store the downloaded file under.
        function: func
            a function that is called to retrieve the DataFrame if the file does not exist.
        path: str (None)
            the path in which the file is stored. If None, it will first check the ~/.pipetorch (for sharing
            dataset between users) and then ~/.pipetorchuser (for user specific caching of datasets).
        save: bool (True)
            whether to save a downloaded .csv
        **kwargs:
            additional parameters passed to pd.read_csv. For example, when a multichar delimiter is used
            you will have to set engine='python'.
            
    Returns: DFrame
    """
    return DFrame(read_pd_from_function(filename, function, path=path, save=save, **kwargs))
    
def read_torchtext(torchtext_function):
    try:
        return torchtext_function(root=path_shared() / torchtext_function.__name__)
    except:
        return torchtext_function(root=path_user() / torchtext_function.__name__)

def wine_quality():
    return read_from_kaggle('uciml/red-wine-quality-cortez-et-al-2009')

def telco_churn():
    return read_from_kaggle('blastchar/telco-customer-churn')

def movielens_ratings():
    COLS = ['user_id', 'movie_id', 'rating', 'timestamp']
    return read_from_kaggle('odedgolden/movielens-1m-dataset', 'ratings.dat', sep='::', engine='python', names=COLS)

def movielens_movies():
    COLS = ['movie_id', 'title', 'genre']
    return read_from_kaggle('odedgolden/movielens-1m-dataset', 'movies.dat', sep='::', engine='python', names=COLS)
    
def movielens_users():
    COLS = ['user_id', 'gender', 'age', 'occupation', 'zip_code']
    return read_from_kaggle('odedgolden/movielens-1m-dataset', 'users.dat', sep='::', engine='python', names=COLS)
    
def dam_outflow():
    return read_from_package('pipetorch', 'data/datasets/dam_outflow.csv')
    
def boston_housing_prices():
    return read_from_kaggle('fedesoriano/the-boston-houseprice-data')

def hotel_booking():
    return read_from_kaggle('mojtaba142/hotel-booking')
    df = df.sort_values(by=['arrival_date_year', 'arrival_date_week_number'])
    df = df[[ c for c in df if c != 'is_canceled'] + ['is_canceled']]
    return DFrame(df)

def hotel():
    df = hotel_booking()
    train = df[(df.ArrivalDateYear < 2017) | (df.ArrivalDateWeekNumber < 14)]
    return DFrame(train)

def hotel_test_orig():
    df = hotel_booking()
    hotel = df[(df.ArrivalDateYear == 2017) & (df.ArrivalDateWeekNumber > 13)]
    return DFrame(hotel)

def hotel_test():
    return hotel_test_orig().drop(columns='IsCanceled')

def hotel_test_y():
    return hotel_test_orig()[['IsCanceled']]

def hotel_test_score(pred_y):
    return f1_score(hotel_test_y(), pred_y)

def iris():
    return read_from_kaggle('uciml/iris')

def bank_marketing():
    return read_from_kaggle('janiobachmann/bank-marketing-dataset')

def auto_mpg():
    return read_from_kaggle('uciml/autompg-dataset')

def big_mart_sales():
    return read_from_kaggle('brijbhushannanda1979/bigmart-sales-data', train='Train.csv', test='Test.csv')

def advertising_channels():
    return read_from_kaggle('yasserh/advertising-sales-dataset')

def titanic():
    read_from_kaggle_competition('titanic', 'train.csv', 'test.csv')

def diamonds():
    return read_from_kaggle('shivam2503/diamonds')

def indian_liver():
    return read_from_kaggle('uciml/indian-liver-patient-records')

def ames_housing():
    return read_from_kaggle_competition('house-prices-advanced-regression-techniques', 'train.csv', 'test.csv')
    
def air_passengers():
    return read_from_kaggle('rakannimer/air-passengers')

def rossmann_store_sales():
    return read_from_kaggle_competition('rossmann-store-sales', 'train.csv', 'test.csv')

def rossmann_stores():
    return read_from_kaggle_competition('rossmann-store-sales', 'store.csv')

def california():
    return read_from_kaggle('camnugent/california-housing-prices')
    
def heart_disease(**kwargs):
    return read_from_kaggle('johnsmith88/heart-disease-dataset')
    
def speeddate(**kwargs):
    return read_from_kaggle('annavictoria/speed-dating-experiment')
    
def occupancy():
    return read_from_kaggle('robmarkcole/occupancy-detection-data-set-uci', 'datatraining.txt', 'datatest.txt')

def ag_news(language='basic_english', min_freq=1, collate='pad'):
    from torchtext.datasets import AG_NEWS
    train_iter, test_iter = read_torchtext(AG_NEWS)
    tc = TextCollection.from_iter(train_iter, None, test_iter, min_freq=min_freq).collate(collate)
    return tc

def bbc_news(language='basic_english', min_freq=1, collate='pad'):
    return TextCollection.from_csv('/data/datasets/bbc-text.csv', language=language, min_freq=min_freq).collate(collate)
