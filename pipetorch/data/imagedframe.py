from ..data.dframe import DFrame, Databunch
from ..data.kagglereader import Kaggle
from torchvision.datasets import ImageFolder
from torchvision.transforms import transforms
from torch.utils.data import random_split
import torch
import pandas as pd
import numpy as np
import torch.nn.functional as F
import matplotlib.pyplot as plt
from PIL import Image, ImageStat
import random
from pathlib import Path

def LoadImage(fp):
    return Image.open(fp[0])

def LoadImageFast(fp):
    import pyvips
    image = pyvips.Image.new_from_file(fp[0], access="sequential")
    image = image.colourspace("srgb")
    mem_img = image.write_to_memory()
    return np.frombuffer(mem_img, dtype=np.uint8).reshape(image.height, image.width, 3)

def _subplots(rows, cols, imgsize=4, figsize=None, title=None, **kwargs):
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

def _show_batch(ds, rows=3, imgsize=(20,20), figsize=(10,10), classes=None, normalized_mean=None, normalized_std=None):
    inv_normalizer = lambda x:x
    if normalized_std is not None and normalized_mean is not None:
            inv_normalizer = transforms.Normalize(mean=tuple(-m/s for m, s in zip(normalized_mean, normalized_std)), std=tuple(1/s for s in normalized_std))

    axs = _subplots(rows, rows, imgsize=imgsize, figsize=figsize).flatten()

    for i, ax in enumerate(axs):
        img, y = ds[random.randrange(0, len(ds))]
        img = transforms.Resize([100,100])(img)
        img = inv_normalizer(img)
        try:
            img = transforms.ToTensor()(img)
        except: pass
        try:
            img = img.numpy()
#                 if img.shape[0] == 1:
#                     img = img.squeeze(axis=0)
#                 elif img.shape[0] == 3:
#                     img = img.permute(1, 2, 0)
        except: pass
        if img.shape[0] == 1:
            img = img.squeeze(axis=0)
        elif img.shape[0] == 3:
            img = np.transpose(img, (1, 2, 0))
        img = np.clip(img, 0, 1)    
        ax.imshow(img)
        try:
            y = classes[int(y)]
        except:
            raise
        ax.set_title(f'y={y}')
    for ax in axs.flatten()[i:]:
        ax.axis('off')
    plt.tight_layout()
    plt.show()

class ImageDatabunch(Databunch):
    def __init__(self, train_ds, valid_ds=None, test_ds=None, batch_size=32, 
                 valid_batch_size=None, num_workers=2, shuffle=True, pin_memory=False, 
                 balance=False, collate=None, 
                 normalized_mean=None, normalized_std=None, classes=None):
        if valid_batch_size is None:
            valid_batch_size = batch_size
        try:
            self.classes = classes or train_ds.classes
        except:
            try:
                self.classes = classes or train_ds.dataset.classes
            except: pass
        try:
            self.normalized_mean = normalized_mean or train_ds.normalized_mean
            self.normalized_std = normalized_std or train_ds.normalized_std
        except:
            try:
                self.normalized_mean = normalized_mean or train_ds.dataset.normalized_mean
                self.normalized_std = normalized_std or train_ds.dataset.normalized_std
            except: pass
        super().__init__(None, train_ds, valid_ds=valid_ds, test_ds=test_ds, batch_size=batch_size, 
                         valid_batch_size=valid_batch_size, num_workers=num_workers, shuffle=shuffle, 
                         pin_memory=pin_memory, balance=balance, collate=collate)

    @classmethod
    def from_train_test_ds(cls, train_valid_ds, test_ds, valid_perc=0.2, 
                           batch_size=32, valid_batch_size=None, 
                           num_workers=2, shuffle=True, pin_memory=False, balance=False, collate=None,
                           classes=None, normalized_mean=None, normalized_std=None):
        valid_length = int(valid_perc * len(train_valid_ds))
        train_length = len(train_valid_ds) - valid_length
        if shuffle:
            train_ds, valid_ds = random_split(train_valid_ds, [train_length, valid_length])
        else:
            train_ds = train_valid_ds[train_length]
            valid_ds = train_valid_ds[valid_length]
        return cls(train_ds, valid_ds, test_ds, batch_size=batch_size, 
                   valid_batch_size=valid_batch_size, num_workers=num_workers, shuffle=shuffle, 
                   pin_memory=pin_memory, balance=balance, collate=collate, classes=classes,
                   normalized_mean=normalized_mean, normalized_std=normalized_std)
        
    def show_batch(self, rows=3, imgsize=(20,20), figsize=(10,10)):
        try:
            _show_batch( self.train_ds, rows=rows, imgsize=imgsize, figsize=figsize, classes=self.classes,
                       normalized_mean=self.normalized_mean, normalized_std=self.normalized_std)
        except:
            _show_batch( self.train_ds, rows=rows, imgsize=imgsize, figsize=figsize, classes=self.classes)
            

class ImageDFrame(DFrame):
    _metadata = DFrame._metadata + ['_pt_dataset_transforms', '_pt_normalize', '_pt_normalized_mean', '_pt_normalized_std',
                                    '_pt_classes', '_pt_class_to_idx']

    _internal_names = DFrame._internal_names + ['_pt__locked_normalized_mean', '_pt__locked_normalized_std']
    
    _internal_names_set = set( _internal_names )
    
    def __init__(self, data, *args, classes=None, **kwargs):
        super().__init__(data, *args, **kwargs)
        self._pt_dtype = str
        self._pt__locked_normalized_mean = None
        self._pt__locked_normalized_std = None
        self.normalize()
        self._pt_classes = classes
        
    @classmethod
    def read_from_kaggle(cls, dataset, filename=None, shared=False, force=False, **kwargs):
        k = Kaggle(dataset, shared=shared)
        if force:
            k.remove_user()
        folder = k.file(filename)
        subfolders = len([ 1 for file in folder.glob('**/*') if file.is_dir() ])
        if subfolders > 1:
            return cls.from_image_folder(folder)
        else:
            return cls.from_image_files(folder, **kwargs)

    def to_databunch(self, datasetclass=None, batch_size=32, valid_batch_size=None, 
                     num_workers=0, shuffle=True, pin_memory=False, 
                     balance=False, collate=None):
        return ImageDatabunch(*self.to_datasets(datasetclass=datasetclass), 
                         batch_size=batch_size, valid_batch_size=valid_batch_size,
                         num_workers=num_workers, shuffle=shuffle, 
                         pin_memory=pin_memory, balance=balance, collate=collate, classes=self.classes,
                         normalized_mean=self.normalized_mean, normalized_std=self.normalized_std)  
    
    @property
    def _constructor(self):
        return ImageDFrame

    @property
    def classes(self):
        return self._pt_classes
    
    @property
    def class_to_idx(self):
        return self._pt_class_to_idx
    
    @classes.setter
    def classes(self, value):
        if value is not None:
            self._pt_classes = value
            self._pt_class_to_idx = { c:i for i, c in enumerate(value) }
    
    def _dataset_pre_transforms(self):
        return [ LoadImage ]
    
    def _dataset_post_transforms(self):
        return [ transforms.ToTensor() ]
    
    def _train_transformation_parameters(self, train_dset):
        """
        This will be called from DFrame when the datasets are consructed on the train dataset.
        """
        if self._pt_normalize:
            if self._pt_normalized_mean is None or self._pt_normalized_std is None:
                n = 0
                sample = range(len(train_set)) if len(train_dset) < 100 else random.sample(range(len(train_dset)), min(100, len(train_dset)))
                dset = train_dset.to_dataset()
                for i in sample:
                    x, y = dset[i]
                    try:
                        channels
                    except:
                        channels = len(x)
                        x1 = [0.0 for i in range(channels)]
                        x2 = [0.0 for i in range(channels)]

                    for c in range(channels):
                        x1[c] += x[c].view(-1).sum()
                        x2[c] += (x[c].view(-1) ** 2).sum()
                    n = n + x[0].shape[0] * x[0].shape[1]
                sd = np.zeros(channels)
                mean = np.zeros(channels)
                for c in range(channels):
                    sd[c] = np.sqrt(((n * x2[c]) - (x1[c] * x1[c])).numpy() / (n * (n - 1)))
                    mean[c] = (x1[c] * x1[c]).numpy() / (n * (n - 1))
                self._pt__locked_normalized_mean = torch.tensor(mean) 
                self._pt__locked_normalized_std = torch.tensor(sd)
                
            else:
                self._pt__locked_normalized_mean = self._pt_normalized_mean
                self._pt__locked_normalized_std = self._pt_normalized_std
            return True

    def normalize(self, do=True, normalized_mean=None, normalized_std=None):
        """
        Normalize the images. You can either supply normalize with a precalculated mean and standard
        deviation per channel, or pass True to automatically calculate these parameters on the 
        training set. This will require an additional pass over the training set.
        
        Arguments:
            do: bool (True) - if True images are normalized
            
            normalized_mean: (float)
                set of precalculated means for the dataset. 
                The set size should be the same as the number of channels in the images.
                
            normalized_std: (float)
                set of precalculated standard deviations for the dataset
                The set size should be the same as the number of channels in the images.
        """
        self._pt_normalize = do
        self._pt_normalized_mean = torch.tensor(normalized_mean) if normalized_mean is not None else None
        self._pt_normalized_std = torch.tensor(normalized_std) if normalized_std is not None else None
        return self

    @property
    def normalized_mean(self):
        return self._pt__locked_normalized_mean    
        
    @normalized_mean.setter
    def normalized_mean(self, value):
        self._pt__locked_normalized_mean = value
    
    @property
    def normalized_std(self):
        return self._pt__locked_normalized_std
    
    @normalized_std.setter
    def normalized_std(self, value):
        self._pt__locked_normalized_std = value
        
    def _dataset_transforms(self, pre=True, train=True, standard=True, post=True, normalize=True):
        t = super()._dataset_transforms(pre=pre, train=train, standard=standard, post=post)
        if normalize and self._pt__locked_normalized_mean is not None and self._pt__locked_normalized_std is not None:
            t.append(transforms.Normalize(mean=self._pt__locked_normalized_mean, 
                                          std=self._pt__locked_normalized_std))
        return t
    
    def train_images_ds(self):
        """
        returns a version of the train DataSet, for which normalization and post_transforms are turned off
        in other words, this will return the images just before they are converted to tensors. 
        """
        return self.dtype(False)._dset_indices(self._train_indices, self._dataset_transforms(post=False, normalize=False)).to_dataset()
    
    def show_batch(self, rows=3, imgsize=(20,20), figsize=(10,10)):
        """
        Shows a sample of rows*rows images from the training set.
        """
        _show_batch( self.train_images_ds(), rows=rows, imgsize=imgsize, figsize=figsize, classes=self.classes)
    
    @classmethod
    def from_binary_folder(cls, folder, **kwargs):
        """
        Construct an ImageDFrame from the filelist that is obtained from TorchVision's ImageFolder,
        in other words, the subfolders are the class labels assigned to the files they contain.
        
        from_binary_folder assumes that we will attempt to use the dataset with a BCELoss functions
        and put the target variable as float32 in a column (2D) vector 
        """
        try:
            folder.samples
        except:
            folder = ImageFolder(root=folder)
        r = ImageDFrame(folder.samples, columns=['filename', 'target'], classes=folder.classes)
        r.target = r.target.astype(np.int64)
        return r
    
    @classmethod
    def from_image_folder(cls, folder):
        """
        Construct an ImageDFrame from the filelist that is obtained from TorchVision's ImageFolder,
        in other words, the subfolders are the class labels assigned to the files they contain.

        from_image_folder assumes the data is used with a CrossEntropyLoss function
        and puts the target variable as long in a row (1D) vector 
        """
        try:
            folder.samples
        except:
            folder = ImageFolder(root=folder)
        r = ImageDFrame(folder.samples, columns=['filename', 'target'], classes=folder.classes)
        r.target = r.target.astype(np.int64)
        return r

    @classmethod
    def from_image_files(cls, folder, ext=None, omit=None, delimiter='.'):
        """
        Construct an ImageDFrame from a folder with files in which the
        first part of the filename indicates the class label, e.g. horse.1.jpg,
        cow.2.jpg.
        
        Args:
            folder: str or Path
                folder containing the files
            ext: str (None)
                extension of the files that are included as images
            omit: [ str ] (None)
                omit all files that end with any of the give strings
            delimiter: str ('.')
                delimiter used to obtain the class label from the filename.
                By default a '.' is used, in which case horse.1.jpg will use the
                name 'horse' as its class label.
        """
        if type(folder) == str:
            folder = Path(folder)
        if ext:
            files = list(folder.glob(f'*.{ext}'))
        else: 
            files = list(folder.glob('*'))
        if omit is not None:
            for o in omit:
                files = [ f for f in files if not f.endswith(o) ]
        classes = [ ('/' + str(f)).split('/')[-1] for f in files ]
        classes = [ c.split(delimiter)[0] for c in classes ]
        uniqueclasses = list(set(classes))
        str2class = { c:i for i, c in enumerate(uniqueclasses) }
        classes = [ str2class[c] for c in classes ]
        files = [ (str(f), c) for f, c in zip(files, classes) ]
        r = ImageDFrame(files, columns=['filename', 'target'], classes=uniqueclasses)
        r.target = r.target.astype(np.int64)
        return r

