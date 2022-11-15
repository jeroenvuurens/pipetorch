from pathlib import Path
import pandas as pd
import shutil
import os
from io import StringIO
import pkgutil
from zipfile import ZipFile
from collections import Counter

def read_torchtext(torchtext_function):
    try:
        return torchtext_function(root=path_shared() )
    except:
        return torchtext_function(root=path_user() )

def path_user(dataset=None):
    if dataset is not None:
        return Path.home() / '.pipetorchuser' / dataset.split('/')[-1]
    return Path.home() / '.pipetorchuser'

def path_shared(dataset=None):
    if dataset is not None:
        return Path.home() / '.pipetorch' / dataset.split('/')[-1]    
    return Path.home() / '.pipetorch'

# def path_kaggle(dataset=None):
#     if dataset is not None:
#         return Path('../input') / dataset.split('/')[-1]

# def dataset_path(dataset):
#     try:
#         p = path_kaggle(dataset)
#         if p.exists():
#             return p
#     except: pass
#     try:
#         p = path_shared(dataset)
#         if p.exists():
#             return p
#     except: pass
#     return path_user(dataset)

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

def read_excel(path, filename=None, save=True, **kwargs):
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

# def create_kaggle_authentication(username, key):
#     """
#     Writes the authentication by kaggle to a file, so that PipeTorch can use the kaggle api to 
#     download datasets.
    
#     Arguments:
#         username: str
#             your username at kaggle (you do need to register)
#         key: str
#             the token that you have generated on your kaggle account (can be read in the .json file)
#     """
#     if not os.path.exists(Path.home() / '.kaggle'):
#         os.mkdir(Path.home() / '.kaggle')
#         os.chmod(Path.home() / '.kaggle', 0o700)
#     kagglejson = Path.home() / '.kaggle' / 'kaggle.json'
#     if not os.path.exists(kagglejson):
#         with open (kagglejson, "w") as fout:
#             fout.write("{" + f'"username":"{username}", "key":"{key}"' + "}")
#             print(f'kaggle authorization written to {kagglejson}')
#         os.chmod(kagglejson, 0o600)
#     else:
#         print(f'kaggle authorization already exists, check {kagglejson}')

# def kaggle_download(dataset, shared=False):
#     try:
#         from kaggle.api.kaggle_api_extended import KaggleApi
#         api = KaggleApi()
#         api.authenticate()
#     except:
#         print('''
#         Error authenticating kaggle: you need to (1) register at kaggle (2) generate a token/key and 
#         (3) put your user credentials in ~/.kaggle/kaggle.json. You can perform (3) using 
#         pipetorch.data.create_kaggle_authentication(username, key)
#         ''')
#         return
#     path = path_shared(dataset) if shared else dataset_path(dataset)
#     print(f'Downloading {dataset} from kaggle to {path}')
#     api.dataset_download_files(dataset, path=path, unzip=True)

# def kaggle_download_competition(dataset, shared=False):
#     try:
#         from kaggle.api.kaggle_api_extended import KaggleApi
#         api = KaggleApi()
#         api.authenticate()
#     except:
#         print('''
#         Error authenticating kaggle: you need to (1) register at kaggle (2) generate a token/key and 
#         (3) put your user credentials in ~/.kaggle/kaggle.json. You can perform (3) using 
#         pipetorch.data.create_kaggle_authentication(username, key)
#         ''')
#         return
#     path = path_shared(dataset) if shared else dataset_path(dataset)
#     print(f'Downloading {dataset} from kaggle to {path}')
#     api.competition_download_files(dataset, path=path)
#     for zfile in list(path.glob('*.zip')):
#         zip = ZipFile(zfile)
#         zip.extractall(path=path)
#         zfile.unlink()
        
# def read_from_kaggle(dataset, filename=None, shared=False, force=False, **kwargs):
#     if force:
#         try:
#             path = path_user(dataset)
#             shutil.rmtree(path)
#         except: pass
#     path = dataset_path(dataset)
#     if force or not path.exists():
#         kaggle_download(dataset, shared=shared)
#         path = dataset_path(dataset)
#     assert path.exists(), f'Problem (down)loading Kaggle dataset {dataset}'
#     if filename is not None:
#         files = list(path.glob(filename))
#     else:
#         files = list(path.glob('**/*'))
#         if len(files) > 1:
#             ext = Counter([ str(f).split('.')[-1] for f in files ])
#             if ext['csv'] == 1:
#                 files = list(path.glob('**/*.csv'))
#     assert len(files) == 1, f'There are multiple files that match {files}, set filename to select a file'
#     return pd.read_csv(files[0], **kwargs)

# def read_from_kaggle_competition(dataset, filename=None, shared=False, force=False, **kwargs):
#     if force:
#         try:
#             path = path_user(dataset)
#             shutil.rmtree(path)
#         except: pass
#     path = dataset_path(dataset)
#     if force or not path.exists():
#         kaggle_download_competition(dataset, shared=shared)
#     path = dataset_path(dataset)
#     assert path.exists(), f'Problem downloading Kaggle dataset {dataset}'
#     if filename is not None:
#         files = list(path.glob(filename))
#     else:
#         files = list(path.glob('**/*'))
#         if len(files) > 1:
#             ext = Counter([ str(f).split('.')[-1] for f in files ])
#             if ext['csv'] == 1:
#                 files = list(path.glob('**/*.csv'))
#     assert len(files) == 1, f'There are multiple files that match {files}, set filename to select a file'
#     return pd.read_csv(files[0], **kwargs)

# def read_csv(url, filename=None, path=None, save=False, **kwargs):
#     """
#     Reads a .csv file from cache or url. The place to store the file is indicated by path / filename
#     and when a delimiter is used, this is also used to save the file so that the original delimiter is kept.
#     The file is only downloaded using the url if it does not exsists on the filing system. If the file is
#     downloaded and save=True, it is also stored for future use.
    
#     Arguments:
#         url: str
#             the url to download or a full path pointing to a .csv file
#         filename: str (None)
#             the filename to store the downloaded file under. If None, the filename is extracted from the url.
#         path: str (None)
#             the path in which the file is stored. If None, it will first check the ~/.pipetorch (for sharing
#             dataset between users) and then ~/.pipetorchuser (for user specific caching of datasets).
#         save: bool (False)
#             whether to save a downloaded .csv
#         **kwargs:
#             additional parameters passed to pd.read_csv. For example, when a multichar delimiter is used
#             you will have to set engine='python'.
            
#     Returns: pd.DataFrame
#     """
#     if filename is None:
#         filename = get_filename(url)
#     storedpath = get_stored_path(filename, path)        
#     if not storedpath.exists():
#         storedpath = (path_user() / filename)
#         (path_user()).mkdir(exist_ok=True)
#         if '://' in url:
#             print(f'Downloading {url}')
#         df = pd.read_csv(url, **kwargs)
#         if save:
#             print(f'saving to {storedpath}')
#             to_csv(df, filename, **kwargs)
#         return df
#     else:
#         return pd.read_csv(storedpath, **kwargs)
    
def read_from_package(package, filename, **kwargs):
    csv = pkgutil.get_data(package, filename).decode()
    return pd.read_csv(StringIO(csv), **kwargs)

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
            
    Returns: pd.DataFrame
    """
    storedpath = get_stored_path(filename, path)
    if storedpath.is_file():
        return pd.read_csv(storedpath, **kwargs)
    df = function()
    to_csv(df, storedpath, **kwargs)
    return df
