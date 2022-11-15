
from .imagedframe import ImageDatabunch
import torch
import torch.nn.functional as F
from torchvision.datasets import MNIST, CIFAR10, FashionMNIST
from torchvision.transforms import transforms
from pathlib import Path
from getpass import getuser
import pandas as pd
import numpy as np
import os
from tqdm.notebook import tqdm
import io
from PIL import Image, ImageStat
from ..helper import is_documented_by
    
def create_path(p, mode=0o777):
    path = Path(p)
    os.makedirs(path, mode, exist_ok=True)
    return path

def image_folder():
    return f'/tmp/{getuser()}/images'

def _fast_torchvision(func, root=None, train=False, transform=None, validtransform=None, 
                      size=None, download=False, **kwargs):
    """
    A Dataset for MNIST, that caches the dataset on gpu for faster processing. 
    The images dataset are shaped SIZE.
    
    Arguments:
        root: str (None)
            local path where the dataset is stored. If None, first ~/.pipetorchuser is checked, then
            ~/.pipetorch (for shared datasets).
        
        train: bool (True)
            whether the train or test set is retrieved
            
        transform: callable (None)
            an optional function to perform additional transformations on the data
            not that the images are already tensors
            
        size: int (None)
            the required width/height of the image in pixels
            
        normalize: bool (True)
            Whether or not to normalize the images, which you normally should do, 
            but when you wish to inspect some of the images, you can turn it off.
            
        **kwargs: dict
            passed to torchvision's MNIST
    """
    if root is None:
        try:
            return func(root='~/.pipetorchuser', download=False, train=train, 
                        transform=transform, validtransform=validtransform, size=size, **kwargs)
        except: pass
        try:
            return func(root='~/.pipetorch', download=False, train=train, 
                        transform=transform, validtransform=validtransform, size=size, **kwargs)
        except: pass
        root = '~/.pipetorchuser'
    return func(root=root, download=download, train=train, 
                transform=transform, validtransform=validtransform, size=size, **kwargs)

class _FastCIFAR(CIFAR10):
    def __init__(self, root=None, train=True, transform=None, validtransform=None, 
                 normalize=True, size=None, **kwargs):
        super().__init__(root=root, train=train, **kwargs)
        self.transform=self.extend_transform(transform if train else validtransform, size)
        # Scale data to [0,1]
        self.data = torch.tensor(self.data).float().div(255)
        self.data = self.data.permute(0, 3, 1, 2)
        # Normalize it with the usual MNIST mean and std
        if normalize:
            self.normalized_mean = np.array([0.485, 0.456, 0.406 ])
            self.normalized_std = np.array([0.229, 0.224, 0.225 ])
            self.data[:,0].sub_(self.normalized_mean[0]).div_(self.normalized_std[0])
            self.data[:,1].sub_(self.normalized_mean[1]).div_(self.normalized_std[1])
            self.data[:,2].sub_(self.normalized_mean[2]).div_(self.normalized_std[2])
        self.targets = torch.tensor(self.targets)
        
    def extend_transform(self, transform, size=None):
        if size is None:
            return transform
        if transform is None:
            return transforms.Resize(size)
        return transforms.Compose([transforms.Resize(size), transform])
        
    def __getitem__(self, index):
        if self.transform:
            return self.transform(self.data[index]), self.targets[index]
        return self.data[index], self.targets[index]

class _FastFashionMNIST(FashionMNIST):
    def __init__(self, root=None, train=True, transform=None, validtransform=None, 
                 size=None, normalize=True, **kwargs):
        super().__init__(root=root, train=train, **kwargs)
        self.transform=self.extend_transform(transform if train else validtransform, size)
        self.data = self.data.unsqueeze_(dim=1).float().div(255)
        if normalize:
            self.data = self.data.div(0.5).sub(0.5)
            self.normalized_mean = np.array([0.5])
            self.normalized_std = np.array([0.5])

    def extend_transform(self, transform, size=None):
        if size is None:
            return transform
        if transform is None:
            return transforms.Resize(size)
        return transforms.Compose([transforms.Resize(size), transform])
        
    def __getitem__(self, index):
        if self.transform:
            return self.transform(self.data[index]), self.targets[index]
        return self.data[index], self.targets[index]

class _FastMNIST(MNIST):
    def __init__(self, root, train=True, transform=None, validtransform=None,
                 size=None, normalize=True, **kwargs):
        super().__init__(root=root, train=train, **kwargs)
        self.transform=self.extend_transform(transform if train else validtransform, size)

        # Scale data to [0,1]
        self.data = self.data.unsqueeze(1).float().div(255)
        if normalize:
            # Normalize it with the usual MNIST mean and std
            self.data.sub_(0.1307).div_(0.3081)
            self.normalized_mean = np.array([0.1307])
            self.normalized_std = np.array([0.3081])

    def extend_transform(self, transform, size=None):
        if size is None:
            return transform
        if transform is None:
            return transforms.Resize(size)
        return transforms.Compose([transforms.Resize(size), transform])
        
    def __getitem__(self, index):
        if self.transform:
            return self.transform(self.data[index]), self.targets[index]
        return self.data[index], self.targets[index]

class _FastMNIST3(_FastMNIST):
    def __init__(self, root, train=True, transform=None,
                 normalize=True, **kwargs):
        super().__init__(root, train=train, transform=transform, normalize=normalize, **kwargs)
        if normalize:
            self.normalized_mean = np.array([0.1307, 0.1307, 0.1307])
            self.normalized_std = np.array([0.3081, 0.3081, 0.3081, ])
        
    def __getitem__(self, index):
        img, target = self.data[index], self.targets[index]
        if self.transform:
            img = self.transform(img)
        img = torch.cat([img, img, img], axis=0)
        return img, target
    
@is_documented_by(_fast_torchvision, {'MNIST':'CIFAR10', 'SIZE':'(3,32,32)'})
def FastCIFAR(root=None, train=False, transform=None, validtransform=None, 
              normalize=True, download=False, **kwargs):
    return _fast_torchvision(_FastCIFAR, root=root, download=download, train=train, transform=transform, 
                             validtransform=validtransform, normalize=normalize, **kwargs)

@is_documented_by(_fast_torchvision, {'SIZE':'(1,28,28)'})
def FastMNIST(root=None, train=True, transform=None, validtransform=None,
              size=None, normalize=True, download=False, **kwargs):
    return _fast_torchvision(_FastMNIST, root=root, download=download, train=train, transform=transform, 
                             validtransform=validtransform, normalize=normalize, **kwargs)
                 
@is_documented_by(_fast_torchvision, {'MNIST':'FashionMNIST', 'SIZE':'(3,28,28)'})
def FastFashionMNIST(root=None, train=True, transform=None, validtransform=None,
                     size=None, download=False, normalize=True, **kwargs):
    return _fast_torchvision(_FastFashionMNIST, root=root, download=download, train=train, transform=transform,
                             validtransform=validtransform, normalize=normalize, **kwargs)
                 
@is_documented_by(_fast_torchvision, {'SIZE':'(3,28,28)'})
def FastMNIST3(root=None, train=True, transform=None, validtransform=None,
               size=None, normalize=True, download=False, **kwargs):
    return _fast_torchvision(_FastMNIST3, root=root, download=download, train=train, transform=transform, 
                             validtransform=validtransform, normalize=normalize, **kwargs)

def _fast_databunch(func, root=None, test=False, valid_perc=0.1, batch_size=64, valid_batch_size=None,
                    transform=None, validtransform=None, size=None, num_workers=2, shuffle=True, 
                    pin_memory=False, balance=False, collate=None, classes=None, 
                    normalized_mean=None, normalized_std=None, **kwargs):
    '''
    An image_databunch of the MNIST dataset, shape is SIZE.
    
    Args:
        test: bool (False)
            if True, the test set is also included in the Databunch
            
        valid_perc: float (0.1)
            the size of the train set that is used for validation
    
        root=None, transform=None, size=None, normalize=True, **kwargs
            see FastMNIST for documentation

        batch_size=64, num_workers=2
            see ImageDatabunch for documentation
            
        **kwargs: dict (None)
            named argument that are passed to the TorchVision function that loads this dataset.
        
    Returns: ImageDatabunch
        An extension to a Databunch that provides additional support for images, like image normalization,
        transformation and show_batch to show an example.    
    '''
    train_ds = func(root, transform=transform, validtransform=validtransform, train=True, size=size, **kwargs)
    if test:
        test_ds = func(root, transform=transform, validtransform=validtransform, train=False, size=size, **kwargs)
    else:
        test_ds = None
    return ImageDatabunch.from_train_test_ds(train_ds, test_ds, valid_perc=valid_perc, batch_size=batch_size, 
                                             valid_batch_size=valid_batch_size, num_workers=num_workers, 
                                             shuffle=shuffle, pin_memory=pin_memory, balance=balance,
                                             collate=collate, classes=classes, normalized_mean=normalized_mean,
                                             normalized_std=normalized_std)

@is_documented_by(_fast_databunch, {'SIZE':'(1,28,28)'}) 
def mnist(root=None, test=False, valid_perc=0.1, batch_size=64, transform=None, validtransform=None, 
          size=None, normalize=True, num_workers=2, **kwargs):
    return _fast_databunch(FastMNIST, root=root, test=test, valid_perc=valid_perc, batch_size=batch_size, 
                           transform=transform, validtransform=validtransform, size=size, 
                           normalize=normalize, num_workers=num_workers, **kwargs)

@is_documented_by(_fast_databunch, {'SIZE':'(3,28,28)'}) 
def mnist3(root=None, test=False, valid_perc=0.1, batch_size=64, transform=None, validtransform=None, 
          size=None, normalize=True, num_workers=2, **kwargs):
    return _fast_databunch(FastMNIST3, root=root, test=test, valid_perc=valid_perc, batch_size=batch_size, 
                           transform=transform, validtransform=validtransform, size=size, 
                           normalize=normalize, num_workers=num_workers, **kwargs)

@is_documented_by(_fast_databunch, {'MNIST':'CIFAR10','SIZE':'(3,28,28)'}) 
def cifar(root=None, test=False, valid_perc=0.1, batch_size=64, transform=None, validtransform=None, 
          size=None, normalize=True, num_workers=2, **kwargs):
    return _fast_databunch(FastCIFAR, root=root, test=test, valid_perc=valid_perc, batch_size=batch_size, 
                           transform=transform, validtransform=validtransform, size=size, 
                           normalize=normalize, num_workers=num_workers, **kwargs)

@is_documented_by(_fast_databunch, {'MNIST':'FashionMNIST','SIZE':'(3,28,28)'}) 
def fashionmnist(root=None, test=False, valid_perc=0.1, batch_size=64, transform=None, validtransform=None, 
          size=None, normalize=True, num_workers=2, **kwargs):
    return _fast_databunch(FastFashionMNIST, root=root, test=test, valid_perc=valid_perc, batch_size=batch_size, 
                           transform=transform, validtransform=validtransform, size=size, 
                           normalize=normalize, num_workers=num_workers, **kwargs)

def _gis_args(keywords, output_directory=None, 
                 image_directory=None, limit=200, format='jpg', color_type='full-color', 
                 size='medium', type='photo', delay=0, **kwargs):
    """
    helper function for crawl_images, which uses google image search
    """
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
    for info on the arguments.
    The supporting library frequently breaks, so do not expect this to work.
    
    Arguments:
        keywords: the keywords passed to google image search to retrieve images
        limit: maximum number of images to retrieve (default=200). You will actually receive less iamges because many links will not work
        output_directory: base folder for the downloads (default: /tmp/username/images/)
        image_directory: subpath to store the images for this query (by default uses the query name)
        format: compression type of photos that are downloaded (default='jpg')
        color-type: default='full-color', see https://google-images-download.readthedocs.io/en/latest/arguments.html
        size: default='medium', see https://google-images-download.readthedocs.io/en/latest/arguments.html
        type: default='photo', see https://google-images-download.readthedocs.io/en/latest/arguments.html
        delay: default=0, to pause between downloads, see https://google-images-download.readthedocs.io/en/latest/arguments.html
        **kwargs: any additional arguments that google-images-download accepts.
    """
    try:
        from .google_images_download import googleimagesdownload
    except:
        raise NotImplemented('Need google images download for this')
    kwargs = _gis_args(keywords, output_directory=output_directory, image_directory=image_directory, 
             limit=limit, format=format, color_type=color_type, size=size, type=type, delay=delay, 
             **kwargs)
    response = googleimagesdownload()   #class instantiation
    paths = response.download(kwargs)   #passing the arguments to the function
    
def filter_images(keywords, folder=None, columns=4, height=200, width=200):
    """
    Removes duplicate images and shows the remaining images so that the user can manually select
    images to remove from the folder by pressing the DELETE button below.
    
    Arguments:
        keywords: subfolder of 'folder' in which the images are stored
        folder: folder/output_directory where the crawled images are stored (e.g. /tmp/username/images)
        columns (4): number of images displayed per row
        height (200): height of the images in pixels
        width (200): width of the images in pixels
    """
    import ipywidgets as widgets

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

_ptdatasetslist = [('FastMNist B/W', 'pt.mnist()', 'https://github.com/y0ast/pytorch-snippets/tree/main/fast_mnist'),
            ('FastMNist RGB', 'pt.mnist3()', 'http://yann.lecun.com/exdb/mnist/'),
            ('FastCifar10', 'pt.cifar()', 'https://www.cs.toronto.edu/~kriz/cifar.html')
           ]
datasets = pd.DataFrame(_ptdatasetslist, columns=['dataset', 'method', 'url'])
