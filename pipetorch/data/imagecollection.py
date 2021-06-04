import pandas as pd
import numpy as np
import torch
import math
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, random_split, Subset
import matplotlib.pyplot as plt
from torchvision.datasets import MNIST, ImageFolder, CIFAR10
from torchvision.transforms import transforms
import os
import matplotlib
import matplotlib.patheffects as PathEffects
from IPython.core import pylabtools
from pathlib import Path
import sys
from IPython import get_ipython
from google_images_download import google_images_download
from tqdm.notebook import tqdm
import ipywidgets as widgets
import io
from PIL import Image, ImageStat
from getpass import getuser
from ..evaluate.evaluate import Evaluator

ipython = get_ipython()
back2gui = { b:g for g, b in pylabtools.backends.items() }

class plt_gui:
    def __init__(self, gui):
        self.gui = gui

    def __enter__(self):
        backend = matplotlib.get_backend()
        self.old_gui = back2gui[backend]
        ipython.magic('matplotlib ' + self.gui)

    def __exit__(self, *args):
        ipython.magic('matplotlib ' + self.old_gui)

class plt_inline(plt_gui):
    def __init__(self):
        super().__init__('inline')

class plt_notebook(plt_gui):
    def __init__(self):
        super().__init__('notebook')

def subplots(rows, cols, imgsize=4, figsize=None, title=None, **kwargs):
    "Like `plt.subplots` but with consistent axs shape, `kwargs` passed to `fig.suptitle` with `title`"
    if figsize is None:
        figsize = (imgsize*cols, imgsize*rows)
    fig, axs = plt.subplots(rows,cols,figsize=figsize)
    if rows==cols==1:
        axs = [[axs]]
    elif (rows==1 and cols!=1) or (cols==1 and rows!=1):
        axs = [axs]
    if title is not None:
        fig.suptitle(title, **kwargs)
    return np.array(axs)

def sample(self, device=None):
    X, y = self.one_batch()
    if device is not None:
        return X.to(device), y.to(device)
    return X, y

class ImageDataset(Dataset):
    """Image dataset."""

    def __init__(self, *args, transform=None, **kwargs):
        """
        Args:
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        super.__init__(*args, **kwargs)
        self.transform = transform

    def __getitem__(self, idx):
        item = super.__getitem__(idx)

        if self.transform:
            item = self.transform(item)

        return item

class image_databunch:
    def __init__(self, train_ds, valid_ds, batch_size=32, valid_batch_size=None, shuffle=True, num_workers=0, 
                 pin_memory=False, valid_pin_memory=None, normalized_mean=None, normalized_std=None):
        self.train_ds = train_ds
        self.valid_ds = valid_ds
        self.batch_size = batch_size
        self.valid_batch_size = batch_size if valid_batch_size is None else valid_batch_size
        self.valid_pin_memory = pin_memory if valid_pin_memory is None else valid_pin_memory
        self.num_workers = num_workers
        self.shuffle = shuffle
        self.pin_memory = pin_memory
        self.normalized_mean = normalized_mean
        self.normalized_std = normalized_std

    @staticmethod
    def balance(X, y):
        indices = [np.where(y==l)[0] for l in np.unique(y)]
        classlengths = [len(i) for i in indices]
        n = max(classlengths)
        mask = np.hstack([np.random.choice(i, n-l, replace=True) for l,i in zip(classlengths, indices)])
        indices = np.hstack([mask, range(len(y))])
        return X[indices], y[indices]

    def to(self, device):
        try:
            self.train_ds.data.to(device)
        except: pass
        try:
            self.train_ds.targets.to(device)
        except: pass
        try:
            self.valid_ds.data.to(device)
        except: pass
        try:
            self.valid_ds.targets.to(device)
        except: pass
        self.device=device
        return self

    def cpu(self):
        return self.to(torch.device('cpu'))

    def gpu(self):
        return self.to(torch.device('cuda:0'))

    @property
    def batch_size(self):
        return self._batch_size

    @batch_size.setter
    def batch_size(self, value):
        self._batch_size = min(value, len(self.train_ds))
        self.reset()

    @property
    def num_workers(self):
        return self._num_workers

    @num_workers.setter
    def num_workers(self, value):
        self._num_workers = value
        self.reset()

    def evaluate(self, *metrics):
        #assert len(metrics) > 0, 'You need to provide at least one metric for the evaluation'
        return Evaluator(self, *metrics)

    @property
    def train_dl(self):
        try:
            return self._train_dl
        except:
            self._train_dl = DataLoader(self.train_ds, num_workers=self.num_workers, shuffle=self.shuffle, batch_size=self.batch_size, pin_memory=self.pin_memory)
            return self._train_dl

    @train_dl.setter
    def train_dl(self, dl):
        self._train_dl = dl

    @property
    def valid_dl(self):
        try:
            return self._valid_dl
        except:
            self._valid_dl = DataLoader(self.valid_ds, shuffle=False, num_workers=self.num_workers, batch_size=self.valid_batch_size, pin_memory=self.valid_pin_memory)
            return self._valid_dl

    @valid_dl.setter
    def valid_dl(self, dl):
        self._valid_dl = dl

    @property
    def train_X(self):
        return self.train_ds.data

    @property
    def train_y(self):
        return self.train_ds.targets

    @property
    def valid_X(self):
        return self.valid_ds.data

    @property
    def valid_y(self):
        return self.valid_ds.targets

    @property
    def train_numpy(self):
        return to_numpy(self.train_X), to_numpy(self.train_y)

    @property
    def valid_numpy(self):
        return to_numpy(self.valid_X), to_numpy(self.valid_y)

    def sample(self, device=None):
        X, y = next(iter(self.train_dl))
        if device is not None:
            return X.to(device), y.to(device)
        return X, y

    def reset(self):
        try:
            del self.valid_dl
        except: pass
        try:
            del self._train_dl
        except: pass

    def show_batch(self, rows=3, imgsize=(20,20), figsize=(10,10)):
        with plt_inline():
            old_backend = matplotlib.get_backend()
            Xs, ys = next(iter(self.train_dl))
            Xs = Xs[:rows*rows]
            ys = ys[:rows*rows]
            axs = subplots(rows, rows, imgsize=imgsize, figsize=figsize)
            invnormalize = self.inv_normalize()
            for x,y,ax in zip(Xs, ys, axs.flatten()):
                x = x.cpu()
                x = invnormalize(x)
                #x = (1/(2*2.25)) * x / 0.25 + 0.5
                im = transforms.ToPILImage()(x).convert("RGB")
                im = transforms.Resize([100,100])(im)
                ax.imshow(im)
                ax.set_title(f'y={y}')
            for ax in axs.flatten()[len(Xs):]:
                ax.axis('off')
            plt.tight_layout()
            plt.show()
   
    @classmethod
    def get_transformations_train(cls, size=224, crop_size=None, crop_padding=None, color_jitter=None, rotate=None, do_flip=True, normalize_mean=None, normalize_std=None):
        return cls.get_transformations(size=size, crop_size=crop_size, crop_padding=crop_padding, color_jitter=color_jitter, rotate=rotate, do_flip=do_flip, normalize_mean=normalize_mean, normalize_std=normalize_std)

    @classmethod
    def get_transformations(cls, size=224, crop_size=None, crop_padding=None, color_jitter=None, rotate=None, do_flip=None, normalize_mean=None, normalize_std=None):
        t = []
        if rotate is not None:
            t.append(transforms.RandomRotation(rotate))
        if color_jitter is not None:
            t.append(transforms.ColorJitter((*color_jitter)))
        if crop_size is not None or crop_padding is not None:
            if crop_size is None:
                crop_size = size
            if crop_padding is None:
                crop_padding = 0
            #t.append(Resize(crop_size + 2 * crop_padding))
            t.append(transforms.RandomCrop(crop_size, padding=crop_padding, pad_if_needed=True))
        if size is not None:
            t.append(transforms.Resize([size,size]))
        if do_flip:
            t.append(transforms.RandomHorizontalFlip())
        t.append(transforms.ToTensor())
        if normalize_mean is not None and normalize_std is not None:
            t.append(transforms.Normalize(mean=normalize_mean, std=normalize_std))
        return transforms.Compose( t )

    def inv_normalize(self):
        if self.normalized_std is not None and self.normalized_mean is not None:
            return transforms.Normalize(mean=tuple(-m/s for m, s in zip(self.normalized_mean, self.normalized_std)), std=tuple(1/s for s in self.normalized_std))
        try:
            for l in self.train_ds.transform.transforms:
                if type(l) == transforms.Normalize:
                    return transforms.Normalize(mean=tuple(-m/s for m, s in zip(l.mean, l.std)), std=tuple(1/s for s in l.std))
        except:pass
        try:
            for l in self.train_ds.dataset.transform.transforms:
                if type(l) == transforms.Normalize:
                    return transforms.Normalize(mean=tuple(-m/s for m, s in zip(l.mean, l.std)), std=tuple(1/s for s in l.std))
        except:pass
        
        return lambda x:x

    @staticmethod
    def tensor_ds(ds):
        try:
            ds1 = TransformableDataset(ds, transforms.ToTensor())
            ds1[0][0].shape[0]
            return ds1
        except:
            return ds
    
    @staticmethod
    def channels(ds):
        return tensor_ds(ds)[0][0].shape[0]
    
    @classmethod
    def train_normalize(cls, ds):
        ds = tensor_ds(ds)
        channels = channels(ds)
        total_mean = []
        total_std = []
        for c in range(channels):
            s = torch.cat([X[c].view(-1) for X, y in ds])
            total_mean.append(s.mean())
            total_std.append(s.std())
        return tuple(total_mean), tuple(total_std) 
    
    @classmethod
    def from_image_folder(cls, path, valid_size=0.2, target_transform=None, size=224, crop_size=None, crop_padding=None, color_jitter=None, rotate=None, do_flip=None, normalize_mean=None, normalize_std=None, normalize=False, **kwargs):
        ds = ImageFolder(root=path, target_transform=target_transform)
        split = int((1-valid_size) * len(ds))
        indices = list(range(len(ds)))
        np.random.shuffle(indices)
        train_idx, valid_idx = indices[:split], indices[split:]
        if normalize:
            assert normalize_mean is None and normalize_std is None, 'You cannot set normalize=True and give the mean or std'
            normalize_mean, normalize_std = cls.train_normalize(Subset(ds, train_idx))
        train_transforms = cls.get_transformations_train(size=size, crop_size=crop_size, crop_padding=crop_padding, color_jitter=color_jitter, rotate=rotate, do_flip=do_flip, normalize_mean=normalize_mean, normalize_std=normalize_std)
        valid_transforms = cls.get_transformations(size=size, normalize_mean=normalize_mean, normalize_std=normalize_std)
        train_ds = TransformableDataset(Subset(ds, train_idx), train_transforms)
        valid_ds = TransformableDataset(Subset(ds, valid_idx), valid_transforms)
        return cls(train_ds, valid_ds, **kwargs)

    @classmethod
    def from_image_folders(cls, trainpath, validpath, size=None, transform=None, target_transform=None, **kwargs):
        if type(transform) is int:
            train_transforms = cls.get_transformations_train(size=transform)
            valid_transforms = cls.get_transformations(size=transform)
        elif type(transform) is dict:
            train_transforms = cls.get_transformations_train(**transform)
            valid_transforms = cls.get_transformations(**transform)
        elif type(transform) is tuple:
            train_transforms, valid_transforms = transform
        elif transform is None:
            train_transforms = transforms.Compose( [transforms.ToTensor()] )
            valid_transforms = train_transforms
        else:
            train_transforms = transform
            valid_transforms = transform
 
        train_ds = ImageFolder(root=trainpath, transform=train_transforms, target_transform=target_transform)
        valid_ds = ImageFolder(root=validpath, transform=valid_transforms, target_transform=target_transform)
        return cls(train_ds, valid_ds, **kwargs)

class TransformableDataset(Dataset):
    def __init__(self, dataset, transform=None):
        self.dataset = dataset
        self.transform = transform

    def __getitem__(self, index):
        x, y = self.dataset[index]
        if self.transform:
            x = self.transform(x)
        return x, y
    
    def __len__(self):
        return len(self.dataset)    
    
class Resize(object):
    def __init__(self, size, interpolation=Image.BILINEAR):
        self.size = size
        self.interpolation = interpolation

    def __call__(self, img):
        old_size = img.size  # old_size[0] is in (width, height) format

        ratio = float(self.size)/min(old_size)
        new_size = tuple([int(x * ratio) for x in old_size])

        return img.resize(new_size, resample=self.interpolation)
    
class FastCIFAR(CIFAR10):
    def __init__(self, root='/data/datasets/cifarnew/', train=True, transform=None, device=None, size=None, **kwargs):
        super().__init__(root=root, train=train, **kwargs)
        self.transform=transform
        # Scale data to [0,1]
        self.data = torch.tensor(self.data).float().div(255)
        self.data = self.data.permute(0, 3, 1, 2)
        if size is not None:
            self.data = F.interpolate(self.data, (3, size, size))
        # Normalize it with the usual MNIST mean and std
        self.data[:,0] = self.data[:,0].sub_(0.4057).div_(0.2039)
        self.data[:,1] = self.data[:,1].sub_(0.5112).div_(0.2372)
        self.data[:,2] = self.data[:,2].sub_(0.5245).div_(0.3238)
        self.targets = torch.tensor(self.targets)
        # Put both data and targets on GPU in advance
        if device is not None:
            self.data, self.targets = self.data.to(device), self.targets.to(device)

    def __getitem__(self, index):
        img, target = self.data[index], self.targets[index]

        if self.transform:
            img = self.transform(img)

        return img, target  
    
class FastMNIST(MNIST):
    def __init__(self, *args, transform=None, device=torch.device('cuda:0'), size=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.transform=transform
        # Scale data to [0,1]
        self.data = self.data.unsqueeze(1).float().div(255)
        if size is not None:
            self.data = F.interpolate(self.data, (size, size))
        # Normalize it with the usual MNIST mean and std
        self.data = self.data.sub_(0.1307).div_(0.3081)
        # Put both data and targets on GPU in advance
        if device is not None:
            self.data, self.targets = self.data.to(device), self.targets.to(device)

    def __getitem__(self, index):
        img, target = self.data[index], self.targets[index]

        if self.transform:
            img = self.transform(img)

        return img, target

class FastMNIST3(FastMNIST):
    def __init__(self, *args, transform=None, device=torch.device('cuda:0'), size=None, **kwargs):
        super().__init__(*args, transform=None, device=torch.device('cuda:0'), **kwargs)
        self.size = size

    def __getitem__(self, index):
        img, target = self.data[index], self.targets[index]
        if self.size is not None:
            img = F.interpolate(img.unsqueeze(0), (self.size, self.size)).squeeze(0)
        if self.transform:
            img = self.transform(img)
        img = torch.cat([img, img, img], axis=0)
        return img, target

def mnist(path='/data/datasets/mnist2', batch_size=64, transform=None, size=None, **kwargs):
    train_ds = FastMNIST(path, transform=transform, train=True, size=size, **kwargs)
    valid_ds = FastMNIST(path, transform=transform, train=False, size=size, **kwargs)
    db = image_databunch(train_ds, valid_ds, batch_size=batch_size, 
                         normalized_mean=(0.1307,), normalized_std=(0.3081,))
    return db

def mnist3(path='/data/datasets/mnist2', batch_size=64, size=None, transform=None, **kwargs):
    train_ds = FastMNIST3(path, transform=transform, train=True, size=size, **kwargs)
    valid_ds = FastMNIST3(path, transform=transform, train=False, size=size, **kwargs)
    db = image_databunch(train_ds, valid_ds, batch_size=batch_size, 
                         normalized_mean=(0.1307, 0.1307, 0.1307), normalized_std=(0.3081, 0.3081, 0.3081))
    return db

def cifar(path='/data/datasets/cifarnew/', batch_size=64, size=None, transform=None, **kwargs):
    train_ds = FastCIFAR(root=path, transform=transform, train=True, size=size, **kwargs)
    valid_ds = FastCIFAR(root=path, transform=transform, train=False, size=size, **kwargs)
    db = image_databunch(train_ds, valid_ds, batch_size=batch_size, 
                         normalized_mean=(0.4057, 0.5112, 0.5245), normalized_std=(0.2039, 0.2372, 0.3238))
    return db

def create_path(p, mode=0o777):
    path = Path(p)
    os.makedirs(path, mode, exist_ok=True)
    return path

def image_folder():
    return f'/tmp/{getuser()}/images'

def _gis_args(keywords, output_directory=None, 
                 image_directory=None, limit=200, format='jpg', color_type='full-color', 
                 size='medium', type='photo', delay=0, **kwargs):
    if output_directory is None:
        output_directory = str(create_path(image_folder()))
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
        folder = Path(image_folder())
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

