from torchvision.models import *
import torch
import torch.nn as nn
import torch.nn.functional as F

def split_size(size):
    if type(size) is list or type(size) is tuple:
        return size
    return size, size

def compute_size(size, kernel, stride, padding, layer):
    s = (size - kernel + 2 * padding) / stride + 1
    assert s == int(s), f'size {size} at layer {layer} does not match with kernel size {kernel}, stride {stride} and padding {padding}'
    return int(s)

def grab_r_channel(x):
    return  x[:,0,:,:].view(x.shape[0], 1, x.shape[2], x.shape[3])

class ConvNet_Regression(nn.Module):
    def __init__(self, *layers, size=224, kernel_size=3, stride=1, linear=[], padding=None, pool_size=2, pool_stride=2, preprocess=None, batchnorm=False, num_classes=1, final_activation=lambda x:x, dropout=0):
        super().__init__()
        self.preprocess = preprocess
        if padding is None:
            padding = kernel_size // 2
        hpixels, vpixels = split_size(size)
        self.layers = []
        self.linears = []
        for i, o in zip(layers[:-1], layers[1:]):
            if batchnorm:
                layer = nn.Sequential( 
                    nn.Conv2d(i, o, kernel_size=kernel_size, stride=stride, padding=padding),
                    nn.BatchNorm2d(o),
                    nn.ReLU(),
                    nn.MaxPool2d(kernel_size=pool_size, stride=pool_stride))
            else:
                layer = nn.Sequential( 
                    nn.Conv2d(i, o, kernel_size=kernel_size, stride=stride, padding=padding),
                    nn.ReLU(),
                    nn.MaxPool2d(kernel_size=pool_size, stride=pool_stride))
            n = self._add_layer(layer)
            self.layers.append(layer)
            hpixels = compute_size(hpixels, kernel_size, stride, padding, f'horizontal conv{n}')
            vpixels = compute_size(vpixels, kernel_size, stride, padding, f'vertical conv{n}')
            hpixels = compute_size(hpixels, pool_size, pool_stride, 0, f'horizontal pool{n}')
            vpixels = compute_size(vpixels, pool_size, pool_stride, 0, f'vertical pool{n}')
        nodes = o * hpixels * vpixels
        for i in linear:
            layer = nn.Sequential( nn.Linear(nodes, i), nn.ReLU() )
            self._add_layer( layer )
            self.linears.append( layer )
            nodes = i
        
        layer = nn.Linear(nodes, num_classes)
        self._add_layer( layer )
        self.linears.append( layer )
        self.fa = final_activation

    def _add_layer( self, layer ):
        n = len(self.layers) + len(self.linears) + 1
        self.__setattr__(f'layer_{n+1}', layer)
        return n
        
    def forward(self, x):
        if self.preprocess is not None:
            x = self.preprocess(x)
        for layer in self.layers:
            x = layer(x)
        x = x.reshape(x.size(0), -1)
        
        for layer in self.linears:
            x = layer(x)
        return self.fa(x)

class ConvNet(ConvNet_Regression):
    """
    Standard Convolution Neural Network
    
    Arguments:
        *layers: int, list of ints
            configures the number of convolutional layers by providing the number of filters each layer has
            
        size: int (224)
            size of each input, they should be square
            
        kernel_size: int
            size of the kernel of each convolutional filter. The use of this model is restricted to kernels 
            of the same size.
    
    """
    def __init__(self, *layers, size=224, kernel_size=3, stride=1, linear=[], padding=None, pool_size=2, pool_stride=2, preprocess=None, batchnorm=False, num_classes=None, final_activation=lambda x:x, dropout=0):
        assert num_classes is not None, 'You have to set num_classes to a value'
        super().__init__(*layers, size=size, kernel_size=kernel_size, stride=stride, linear=linear, padding=padding, pool_size=pool_size, 
                         pool_stride=pool_stride, preprocess=preprocess, batchnorm=batchnorm, dropout=dropout,
                         num_classes=num_classes, final_activation=final_activation)
    
    def post_forward(self, y):
        return torch.argmax(y, axis=1)

class ConvNet_BCE(ConvNet):
    def __init__(self, *layers, size=224, kernel_size=3, stride=1, padding=None, pool_size=2, pool_stride=2, preprocess=None, batchnorm=False, dropout=0):
        super().__init__(*layers, size=size, kernel_size=kernel_size, stride=stride, padding=padding, pool_size=pool_size, 
                         pool_stride=pool_stride, preprocess=preprocess, batchnorm=batchnorm, dropout=dropout,
                         num_classes=1, final_activation=self.final_activation)
        
    def final_activation(self, y):
        return torch.sigmoid(y)
    
    def post_forward(self, y):
        return torch.round(y) 

    
class ConvNet_MNist(ConvNet):
    def __init__(self, *layers, size=28, kernel_size=3, stride=1, padding=None, pool_size=2, pool_stride=2, preprocess=None, batchnorm=False, dropout=0):
        super().__init__(*layers, size=size, kernel_size=kernel_size, stride=stride, padding=padding, pool_size=pool_size, 
                         pool_stride=pool_stride, preprocess=preprocess, batchnorm=batchnorm, dropout=dropout,
                         num_classes=10)


