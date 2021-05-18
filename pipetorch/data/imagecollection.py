import pandas as pd
import numpy as np
import torch
import math
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, random_split
import matplotlib.pyplot as plt
from torchvision.datasets import MNIST, ImageFolder
from torchvision.transforms import transforms
import os
import matplotlib
import matplotlib.patheffects as PathEffects
from IPython.core import pylabtools
from pathlib2 import Path
import sys
from IPython import get_ipython
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
    def __init__(self, train_ds, valid_ds, batch_size=32, valid_batch_size=None, shuffle=True, num_workers=0, pin_memory=False, valid_pin_memory=None, device=torch.device('cuda:0')):
        self.train_ds = train_ds
        self.valid_ds = valid_ds
        self.batch_size = batch_size
        self.valid_batch_size = batch_size if valid_batch_size is None else valid_batch_size
        self.valid_pin_memory = pin_memory if valid_pin_memory is None else valid_pin_memory
        self.num_workers = num_workers
        self.shuffle = shuffle
        self.pin_memory = pin_memory
        self.to( device )

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
    def get_transformations(cls, size=224, do_flip=True):
        t = []
        if size is not None:
            t.append(transforms.Resize([size,size]))
        if do_flip:
            t.append(transforms.RandomHorizontalFlip())
        t.append(transforms.ToTensor())
        t.append(transforms.Normalize(mean=(0.5,0.5,0.5), std=(0.5,0.5,0.5)))
        return transforms.Compose( t )

    def inv_normalize(self):
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

    @classmethod
    def from_image_folder(cls, path, size=None, transforms=None, valid_size=0.2, **kwargs):
        if transforms is None:
            if size is None:
                size=224
            transforms = cls.get_transformations(size=size)
        else:
            assert size is None, 'Specify size through get_transforms'
        ds = ImageFolder(root=path, transform=transforms)
        valid_len = int(valid_size * len(ds))
        train_len = len(ds) - valid_len
        train_ds, valid_ds = random_split(ds, [train_len, valid_len])
        return cls(train_ds, valid_ds, **kwargs)

    @classmethod
    def from_image_folders(cls, trainpath, validpath, size=None, transforms=None, **kwargs):
        if transforms is None:
            if size is None:
                size=224
            transforms = cls.get_transformations(size=size)
        else:
            assert size is None, 'Specify size through get_transforms'
        train_ds = ImageFolder(root=trainpath, transform=transforms)
        valid_ds = ImageFolder(root=validpath, transform=transforms)
        return cls(train_ds, valid_ds, **kwargs)

class FastMNIST(MNIST):
    def __init__(self, *args, transform=None, device=torch.device('cuda:0'), size=None, **kwargs):
        super().__init__(*args, **kwargs)

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

class FastMNIST3(MNIST):
    def __init__(self, *args, transform=None, device=None, size=None, **kwargs):
        super().__init__(*args, **kwargs)

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
        img = torch.cat([img, img, img], axis=0)

        return img, target

def mnist(path='/data/datasets/mnist2', num_workers=0, batch_size=64, transform=None, **kwargs):
    train_ds = FastMNIST(path, transform=transform, train=True, **kwargs)
    valid_ds = FastMNIST(path, transform=transform, train=False, **kwargs)
    db = image_databunch(train_ds, valid_ds, num_workers=num_workers, batch_size=batch_size)
    return db

def mnist3(path='/data/datasets/mnist2', num_workers=0, batch_size=64, transform=None, **kwargs):
    train_ds = FastMNIST3(path, transform=transform, train=True, **kwargs)
    valid_ds = FastMNIST3(path, transform=transform, train=False, **kwargs)
    db = image_databunch(train_ds, valid_ds, num_workers=num_workers, batch_size=batch_size)
    return db

def mnist3old(path='/data/datasets/mnist', num_workers=0, batch_size=64, size=28, **kwargs):
    return image_databunch.from_image_folder(path, transforms=image_databunch.get_transformations(do_flip=False, size=size), num_workers=num_workers, **kwargs)

def cifar(path='/data/datasets/mnist', num_workers=0, batch_size=64, **kwargs):
    return image_databunch.from_image_folder(path, transforms=image_databunch.get_transformations(do_flip=False, size=28), num_workers=num_workers, **kwargs)