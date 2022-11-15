from pathlib import Path
import pandas as pd
import shutil
import os
from io import StringIO
import pkgutil
from zipfile import ZipFile
from collections import Counter

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
        os.mkdir(Path.home() / '.kaggle')
        os.chmod(Path.home() / '.kaggle', 0o700)
    kagglejson = Path.home() / '.kaggle' / 'kaggle.json'
    if not os.path.exists(kagglejson):
        with open (kagglejson, "w") as fout:
            fout.write("{" + f'"username":"{username}", "key":"{key}"' + "}")
            print(f'kaggle authorization written to {kagglejson}')
        os.chmod(kagglejson, 0o600)
    else:
        print(f'kaggle authorization already exists, check {kagglejson}')

class Kaggle:
    def __init__(self, url=None, shared=True):
        self.url = url
        self.shared = shared
        self.competition = (url is None) or ('/' not in url)

    @property
    def api(self):
        try:
            return self._api
        except:
            try:
                from kaggle.api.kaggle_api_extended import KaggleApi
                self._api = KaggleApi()
                self._api.authenticate()
                return self._api
            except:
                raise ValueError('''
                Error authenticating kaggle you need to:
                (1) register at kaggle 
                (2) generate a token/key, and 
                (3) put your user credentials in ~/.kaggle/kaggle.json. 
                
                You can perform (3) using pipetorch.data.create_kaggle_authentication(username, key)
                ''')

    @property
    def path_user(self):
        if self.url is not None:
            return Path.home() / '.pipetorchuser' / self.url.split('/')[-1]
        return Path.home() / '.pipetorchuser'

    @property
    def path_shared(self):
        if self.url is not None:
            return Path.home() / '.pipetorch' / self.url.split('/')[-1]    
        return Path.home() / '.pipetorch'

    @property
    def path_kaggle(self):
        return Path('../input') / self.url.split('/')[-1]
        
    @property
    def path(self):
        try:
            p = self.path_kaggle
            if p.exists():
                return p
        except: pass
        try:
            p = self.path_shared
            if self.shared and p.exists():
                return p
        except: pass
        return self.path_user

    def remove_user(self):
        try:
            shutil.rmtree(self.path_user)
        except: pass 
    
    def download(self):
        if self.competition:
            print(f'Downloading competition {self.url} from kaggle to {self.path}')
            self.api.competition_download_files(self.url, path=self.path)
            self.unzip()
        else:
            print(f'Downloading dataset {self.url} from kaggle to {self.path}')
            self.api.dataset_download_files(self.url, path=self.path, unzip=True)

    def unzip(self):
        for zfile in list(self.path.glob('*.zip')):
            zip = ZipFile(zfile)
            zip.extractall(path=self.path)
            zfile.unlink()     
            
    def file(self, filename=None, ext='csv'):
        if not self.path.exists():
            self.download()
        assert self.path.exists(), f'Problem (down)loading Kaggle dataset {self.url}'
        if filename is not None:
            files = list(self.path.glob(filename))
        else:
            files = list(self.path.glob('**/*'))
            if len(files) > 1:
                exts = Counter([ str(f).split('.')[-1] for f in files ])
                if exts['csv'] == 1:
                    files = list(self.path.glob(f'**/*.{ext}'))
        assert len(files) == 1, f'There are multiple files that match {files}, set filename to select a file'
        return files[0]
        
    def read(self, filename=None, ext='csv', **kwargs):
        return pd.read_csv(self.file(filename, ext), **kwargs)
