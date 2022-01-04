from ..data.dframe import DFrame, Databunch
from ..data.transformabledataset import TransformableDataset
from torchvision.datasets import ImageFolder
from torchvision.transforms import transforms
import torch
import pandas as pd
import numpy as np
import torch.nn.functional as F
import matplotlib.pyplot as plt
from PIL import Image, ImageStat

def LoadImage(fp):
    return Image.open(fp[0])

# class image_databunch:
#     def __init__(self, train_ds, valid_ds, batch_size=32, valid_batch_size=None, shuffle=True, num_workers=0, 
#                  pin_memory=False, valid_pin_memory=None, normalized_mean=None, normalized_std=None, 
#                  classes=None, class_to_idx=None):
#         self.train_ds = train_ds
#         self.valid_ds = valid_ds
#         self.batch_size = batch_size
#         self.valid_batch_size = batch_size if valid_batch_size is None else valid_batch_size
#         self.valid_pin_memory = pin_memory if valid_pin_memory is None else valid_pin_memory
#         self.num_workers = num_workers
#         self.shuffle = shuffle
#         self.pin_memory = pin_memory
#         self.normalized_mean = normalized_mean
#         self.normalized_std = normalized_std
#         self.classes = classes
#         self.class_to_idx = class_to_idx

#     @staticmethod
#     def balance(X, y):
#         indices = [np.where(y==l)[0] for l in np.unique(y)]
#         classlengths = [len(i) for i in indices]
#         n = max(classlengths)
#         mask = np.hstack([np.random.choice(i, n-l, replace=True) for l,i in zip(classlengths, indices)])
#         indices = np.hstack([mask, range(len(y))])
#         return X[indices], y[indices]

#     def to(self, device):
#         try:
#             self.train_ds.data.to(device)
#         except: pass
#         try:
#             self.train_ds.targets.to(device)
#         except: pass
#         try:
#             self.valid_ds.data.to(device)
#         except: pass
#         try:
#             self.valid_ds.targets.to(device)
#         except: pass
#         self.device=device
#         return self

#     def cpu(self):
#         return self.to(torch.device('cpu'))

#     def gpu(self):
#         return self.to(torch.device('cuda:0'))

#     @property
#     def batch_size(self):
#         return self._batch_size

#     @batch_size.setter
#     def batch_size(self, value):
#         self._batch_size = min(value, len(self.train_ds))
#         self.reset()

#     @property
#     def num_workers(self):
#         return self._num_workers

#     @num_workers.setter
#     def num_workers(self, value):
#         self._num_workers = value
#         self.reset()

#     def evaluate(self, *metrics):
#         #assert len(metrics) > 0, 'You need to provide at least one metric for the evaluation'
#         return Evaluator(self, *metrics)

#     @property
#     def labels(self):
#         return self._labels
    
#     @property
#     def train_dl(self):
#         try:
#             return self._train_dl
#         except:
#             self._train_dl = DataLoader(self.train_ds, num_workers=self.num_workers, shuffle=self.shuffle, batch_size=self.batch_size, pin_memory=self.pin_memory)
#             return self._train_dl

#     @train_dl.setter
#     def train_dl(self, dl):
#         self._train_dl = dl

#     @property
#     def valid_dl(self):
#         try:
#             return self._valid_dl
#         except:
#             self._valid_dl = DataLoader(self.valid_ds, shuffle=False, num_workers=self.num_workers, batch_size=self.valid_batch_size, pin_memory=self.valid_pin_memory)
#             return self._valid_dl

#     @valid_dl.setter
#     def valid_dl(self, dl):
#         self._valid_dl = dl

#     @property
#     def train_X(self):
#         return self.train_ds.data

#     @property
#     def train_y(self):
#         return self.train_ds.targets

#     @property
#     def valid_X(self):
#         return self.valid_ds.data

#     @property
#     def valid_y(self):
#         return self.valid_ds.targets

#     @property
#     def train_numpy(self):
#         return to_numpy(self.train_X), to_numpy(self.train_y)

#     @property
#     def valid_numpy(self):
#         return to_numpy(self.valid_X), to_numpy(self.valid_y)

#     def sample(self, device=None):
#         X, y = next(iter(self.train_dl))
#         if device is not None:
#             return X.to(device), y.to(device)
#         return X, y

#     def reset(self):
#         try:
#             del self.valid_dl
#         except: pass
#         try:
#             del self._train_dl
#         except: pass

#     def show_batch(self, rows=3, imgsize=(20,20), figsize=(10,10)):
#         #with plt_inline():
#         old_backend = matplotlib.get_backend()
#         Xs, ys = next(iter(self.train_dl))
#         Xs = Xs[:rows*rows]
#         ys = ys[:rows*rows]
#         axs = subplots(rows, rows, imgsize=imgsize, figsize=figsize)
#         invnormalize = self.inv_normalize()
#         for x,y,ax in zip(Xs, ys, axs.flatten()):
#             x = x.cpu()
#             x = invnormalize(x)
#             im = transforms.ToPILImage()(x).convert("RGB")
#             im = transforms.Resize([100,100])(im)
#             ax.imshow(im)
#             try:
#                 y = self.classes[y]
#             except: pass
#             ax.set_title(f'y={y}')
#         for ax in axs.flatten()[len(Xs):]:
#             ax.axis('off')
#         plt.tight_layout()
#         plt.show()
   
#     @classmethod
#     def get_transformations_train(cls, size=224, crop_size=None, crop_padding=None, color_jitter=None, rotate=None, do_flip=True, normalize_mean=None, normalize_std=None):
#         return cls.get_transformations(size=size, crop_size=crop_size, crop_padding=crop_padding, color_jitter=color_jitter, rotate=rotate, do_flip=do_flip, normalize_mean=normalize_mean, normalize_std=normalize_std)

#     @classmethod
#     def get_transformations(cls, size=224, crop_size=None, crop_padding=None, color_jitter=None, rotate=None, do_flip=None, normalize_mean=None, normalize_std=None):
#         t = []
#         if rotate is not None:
#             t.append(transforms.RandomRotation(rotate))
#         if color_jitter is not None:
#             t.append(transforms.ColorJitter(*color_jitter))
#         if crop_size is not None or crop_padding is not None:
#             if crop_size is None:
#                 crop_size = size
#             if crop_padding is None:
#                 crop_padding = 0
#             t.append(transforms.RandomCrop(crop_size, padding=crop_padding, pad_if_needed=True))
#         if size is not None:
#             t.append(transforms.Resize([size,size]))
#         if do_flip:
#             t.append(transforms.RandomHorizontalFlip())
#         t.append(transforms.ToTensor())
#         if normalize_mean is not None and normalize_std is not None:
#             t.append(transforms.Normalize(mean=normalize_mean, std=normalize_std))
#         return transforms.Compose( t )

#     def inv_normalize(self):
#         if self.normalized_std is not None and self.normalized_mean is not None:
#             return transforms.Normalize(mean=tuple(-m/s for m, s in zip(self.normalized_mean, self.normalized_std)), std=tuple(1/s for s in self.normalized_std))
#         try:
#             for l in self.train_ds.transform.transforms:
#                 if type(l) == transforms.Normalize:
#                     return transforms.Normalize(mean=tuple(-m/s for m, s in zip(l.mean, l.std)), std=tuple(1/s for s in l.std))
#         except:pass
#         try:
#             for l in self.train_ds.dataset.transform.transforms:
#                 if type(l) == transforms.Normalize:
#                     return transforms.Normalize(mean=tuple(-m/s for m, s in zip(l.mean, l.std)), std=tuple(1/s for s in l.std))
#         except:pass
        
#         return lambda x:x

#     @staticmethod
#     def tensor_ds(ds):
#         try:
#             ds1 = TransformableDataset(ds, transforms.ToTensor())
#             ds1[0][0].shape[0]
#             return ds1
#         except:
#             return ds
    
#     @staticmethod
#     def channels(ds):
#         return image_databunch.tensor_ds(ds)[0][0].shape[0]
    
#     @classmethod
#     def train_normalize(cls, ds):
#         ds = image_databunch.tensor_ds(ds)
#         channels = image_databunch.channels(ds)
#         total_mean = []
#         total_std = []
#         for c in range(channels):
#             s = torch.cat([X[c].view(-1) for X, y in ds])
#             total_mean.append(s.mean())
#             total_std.append(s.std())
#         return torch.tensor(total_mean), torch.tensor(total_std) 
    
#     @classmethod
#     def from_image_folder(cls, path, valid_size=0.2, target_transform=None, size=224, crop_size=None, crop_padding=None, color_jitter=None, rotate=None, do_flip=None, normalize_mean=None, normalize_std=None, normalize=False, **kwargs):
#         ds = ImageFolder(root=path, target_transform=target_transform)
#         split = int((1-valid_size) * len(ds))
#         indices = list(range(len(ds)))
#         np.random.shuffle(indices)
#         train_idx, valid_idx = indices[:split], indices[split:]
#         if normalize:
#             assert normalize_mean is None and normalize_std is None, 'You cannot set normalize=True and give the mean or std'
#             normalize_mean, normalize_std = cls.train_normalize(Subset(ds, train_idx))
#         train_transforms = cls.get_transformations_train(size=size, crop_size=crop_size, crop_padding=crop_padding, color_jitter=color_jitter, rotate=rotate, do_flip=do_flip, normalize_mean=normalize_mean, normalize_std=normalize_std)
#         valid_transforms = cls.get_transformations(size=size, normalize_mean=normalize_mean, normalize_std=normalize_std)
#         train_ds = TransformableDataset(Subset(ds, train_idx), train_transforms)
#         valid_ds = TransformableDataset(Subset(ds, valid_idx), valid_transforms)
#         return cls(train_ds, valid_ds, classes=ds.classes, class_to_idx=ds.class_to_idx, 
#                    normalized_mean=normalize_mean, normalized_std=normalize_std, **kwargs)

#     @classmethod
#     def from_image_folders(cls, trainpath, validpath, size=None, transform=None, target_transform=None, **kwargs):
#         if type(transform) is int:
#             train_transforms = cls.get_transformations_train(size=transform)
#             valid_transforms = cls.get_transformations(size=transform)
#         elif type(transform) is dict:
#             train_transforms = cls.get_transformations_train(**transform)
#             valid_transforms = cls.get_transformations(**transform)
#         elif type(transform) is tuple:
#             train_transforms, valid_transforms = transform
#         elif transform is None:
#             train_transforms = transforms.Compose( [transforms.ToTensor()] )
#             valid_transforms = train_transforms
#         else:
#             train_transforms = transform
#             valid_transforms = transform
 
#         train_ds = ImageFolder(root=trainpath, transform=train_transforms, target_transform=target_transform)
#         valid_ds = ImageFolder(root=validpath, transform=valid_transforms, target_transform=target_transform)
#         return cls(train_ds, valid_ds, classes=train_ds.classes, class_to_idx=train_ds.class_to_idx, **kwargs)        

class _ShowBatch:
    def _subplots(self, rows, cols, imgsize=4, figsize=None, title=None, **kwargs):
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

    def _show_batch(self, ds, rows=3, imgsize=(20,20), figsize=(10,10)):
        axs = self._subplots(rows, rows, imgsize=imgsize, figsize=figsize).flatten()

        for i, ax in enumerate(axs):
            img, y = ds[i]
            if not issubclass(type(img), Image.Image) and not isinstance(img, Image.Image):
                img = transforms.ToPILImage()(img)
            img = img.convert("RGB")
            img = transforms.Resize([100,100])(img)
            ax.imshow(img)
            try:
                y = self.classes[int(y)]
            except: pass
            ax.set_title(f'y={y}')
        for ax in axs.flatten()[i:]:
            ax.axis('off')
        plt.tight_layout()
        plt.show()

class ImageDatabunch(Databunch, _ShowBatch):
    def __init__(self, train_ds, valid_ds=None, test_ds=None, batch_size=32, 
                 valid_batch_size=None, num_workers=2, shuffle=True, pin_memory=False, balance=False, collate=None):
        if valid_batch_size is None:
            valid_batch_size = batch_size
        super().__init__(None, train_ds, valid_ds=valid_ds, test_ds=test_ds, batch_size=batch_size, 
                         valid_batch_size=valid_batch_size, num_workers=num_workers, shuffle=shuffle, 
                         pin_memory=pin_memory, balance=balance, collate=collate)

    def show_batch(self, rows=3, imgsize=(20,20), figsize=(10,10)):
        super()._show_batch( self.train_ds, rows=rows, imgsize=imgsize, figsize=figsize)

class ImageDFrame(DFrame, _ShowBatch):
    _metadata = DFrame._metadata + ['_pt_transforms', '_pt_normalize', '_pt_normalize_mean', '_pt_normalize_std',
                                    '_pt_classes', '_pt_class_to_idx']

    _internal_names = DFrame._internal_names + ['_pt__locked_normalize_mean', '_pt__locked_normalize_std']
    
    _internal_names_set = set( _internal_names )
    
    def __init__(self, data, *args, **kwargs):
        super().__init__(data, *args, **kwargs)
        self._pt_dtype = str
        self._pt__locked_normalize_mean = None
        self._pt__locked_normalize_std = None
        self.normalize()
        
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
        self._pt_classes = value
        self._pt_class_to_idx = { c:i for i, c in enumerate(value) }
    
    def _pre_transforms(self):
        return [ LoadImage ]
    
    def _post_transforms(self):
        return [ transforms.ToTensor() ]
    
    def _train_transformation_parameters(self, train_dset):
        if self._pt_normalize:
            if self._pt_normalize_mean is None or self._pt_normalize_std is None:
                n = 0
                for x, y in iter(train_dset.to_dataset()):
                    try:
                        channels
                    except:
                        channels = len(x)
                        x1 = [0.0 for i in range(channels)]
                        x2 = [0.0 for i in range(channels)]

                    for c in range(channels):
                        x1[c] += x[c].view(-1).sum()
                        x2[c] += (x[c].view(-1) ** 2).sum()
                    n = n + len(x[0].view(-1))
                sd = np.zeros(channels)
                mean = np.zeros(channels)
                for c in range(channels):
                    sd[c] = np.sqrt(((n * x2[c]) - (x1[c] * x1[c])) / (n * (n - 1)))
                    mean[c] = (x1[c] * x1[c]) / (n * (n - 1))
                self._pt__locked_normalize_mean = torch.tensor(mean) 
                self._pt__locked_normalize_std = torch.tensor(sd)
            else:
                self._pt__locked_normalize_mean = self._pt_normalize_mean
                self._pt__locked_normalize_std = self._pt_normalize_std
            return True

    def normalize(self, do=True, normalize_mean=None, normalize_std=None):
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
        self._pt_normalize_mean = torch.tensor(normalize_mean) if normalize_mean is not None else None
        self._pt_normalize_std = torch.tensor(normalize_std) if normalize_std is not None else None
        return self

    @property
    def normalize_mean(self):
        return self._pt__locked_normalize_mean    
        
    @property
    def normalize_std(self):
        return self._pt__locked_normalize_std    
        
    def _transforms(self, pre=True, train=True, standard=True, post=True, normalize=True):
        t = super()._transforms(pre=pre, train=train, standard=standard, post=post)
        if normalize and self._pt__locked_normalize_mean is not None and self._pt__locked_normalize_std is not None:
            t.append(transforms.Normalize(mean=self._pt__locked_normalize_mean, 
                                          std=self._pt__locked_normalize_std))
        return t
    
    def train_images_ds(self):
        """
        returns a version of the train DataSet, for which normalization and post_transforms are turned off
        in other words, this will return the images just before they are converted to tensors. 
        """
        return self._dset_indices(self._train_indices, self._transforms(post=False, normalize=False)).to_dataset()
    
    def show_batch(self, rows=3, imgsize=(20,20), figsize=(10,10)):
        """
        Shows a sample of rows*rows images from the training set.
        """
        super()._show_batch( self.train_images_ds(), rows=rows, imgsize=imgsize, figsize=figsize)
    
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
        r = ImageDFrame(folder.samples, columns=['filename', 'target'])
        r.target = r.target.astype(np.float32)
        r.classes = folder.classes
        return r
    
    @classmethod
    def from_image_folder(cls, folder, **kwargs):
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
        r = ImageDFrame(folder.samples, columns=['filename', 'target'])
        r.target = r.target.astype(np.long)
        r.classes = folder.classes
        return r.columny(transpose=True)

