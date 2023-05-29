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
