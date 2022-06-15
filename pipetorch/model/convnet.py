#from torchvision.models import *
import torch
import torch.nn as nn
#import torch.nn.functional as F

def split_int_tuple(value, hdefault=None, vdefault=None):
    try:
        return value[0], value[1]
    except:
        if value is None:
            return hdefault, vdefault
        return value, value

def compute_size(size, kernel, stride, padding, layer):
    s = (size - kernel + 2 * padding) / stride + 1
    assert s == int(s), f'size {size} at layer {layer} does not match with kernel size {kernel}, stride {stride} and padding {padding}'
    return int(s)

def greyscale(x):
    return  x[:,0,:,:].view(x.shape[0], 1, x.shape[2], x.shape[3])

class ConvNet(nn.Module):
    """
    Construct a Convolutional Network based on the given parameters. The networks is always built up as
    a sequence of Convolutional layers, followed by a sequence of Linear layers. The number of pixels 
    at the end of the convolutional layer is automatically computed, and the final layer is automatically added.

    Args:
        *layers: int
            one or more channel sizes that are used for the convolutional layers. e.g for RGB images (3 channels)
            that are processed by two convolutional layers of resp. 32 and 64 filters, set layers to 3, 32, 64.
            
        size: int or (int, int) (default 224)
            the height and width of the images. When squared images are used, one number is enough.
            
        kernel_size: int or (int, int) (default 3)
            the height and width of the kernel used in the nn.Conv2d layers.
            
        stride: int or (int, int) (default 1)
            the stride used in the nn.Conv2d layers
            
        padding: int or (int, int) (default None)
            the padding used in de nn.Conv2d layers, None means kern_size // 2 is used
            
        pool_size: int or (int, int) (default 2)
            the kernel size used in nn.MaxPool2d
            
        pool_stride: int or (int, int) (default 2)
            the stride used in nn.MaxPool2d
            
        preprocess: func (None)
            a function that is called on X prior to feeding the input to the first layer.
            
        batchnorm: bool (False)
            if True, then nn.BatchNorm2d are added to every convolutional layer
            
        dropout: float (False)
            if True, then nn.Dropout2d are added at the end of every convolutional layer.
            Atm this cannot be combined with batchnorm (then dropout is not used)
            
        linear: int or [int] ([])
            the convolutional network is always finished by one or more linear layers. The default adds
            a single linear layer, automatically computing the number of pixels after the last convolutional 
            layer, no information is needed except num_classes. Alternatively, a single int adds two linear 
            layers where the int is the hidden size. Or when linear is a list, the list contains the hidden
            layer sizes of consecutive hidden layers.
            
        num_classes: int (default 1)
            the number of outputs
            
        final_activation: func (None)
            the activation function on the ouput layer. If None and num_classes == 1, binary classification is
            assumed and a nn.Sigmoid is added, otherwise a multi-label classification is assumed and no
            final activation function is added. You can override by providing a final activation function, or
            if no activation is needed with num_classes==1 pass final_activation=lambda x:x
            
        post_forward: func (None)
            The PipeTorch trainer uses the post_forward function on a model to process the results after
            the loss function and before the evaluation. Typical and default use for binary classification 
            is to use torch.round() on the estimated likelihoods and for multi-label classification to use 
            torch.argmax(). To override the default, pass a function.
    """
    
    def __init__(self, *layers, size=224, kernel_size=3, stride=1, linear=[], padding=None, pool_size=2, 
                 pool_stride=2, preprocess=None, batchnorm=False, dropout=None, num_classes=1, final_activation=None,
                 post_forward=None):
        super().__init__()
        self.preprocess = preprocess
        hstride, vstride = split_int_tuple(stride)
        hkernel, vkernel = split_int_tuple(kernel_size)
        hpadding, vpadding = split_int_tuple(padding, hkernel // 2, vkernel // 2)
        hpixels, vpixels = split_int_tuple(size)
        hpool_size, vpool_size = split_int_tuple(pool_size)
        hpool_stride, vpool_stride = split_int_tuple(pool_stride)
        self.layers = []
        self.linears = []
        for i, o in zip(layers[:-1], layers[1:]):
            if batchnorm:
                layer = nn.Sequential( 
                    nn.Conv2d(i, o, kernel_size=(hkernel, vkernel), stride=(hstride, vstride), padding=(hpadding, vpadding)),
                    nn.BatchNorm2d(o),
                    nn.ReLU(),
                    nn.MaxPool2d(kernel_size=(hpool_size, vpool_size), stride=(hpool_stride, vpool_stride)))
            elif dropout:
                layer = nn.Sequential( 
                    nn.Conv2d(i, o, kernel_size=(hkernel, vkernel), stride=(hstride, vstride), padding=(hpadding, vpadding)),
                    nn.ReLU(),
                    nn.MaxPool2d(kernel_size=(hpool_size, vpool_size), stride=(hpool_stride, vpool_stride)),
                    nn.Dropout2d(p=dropout)
                )
            else:
                layer = nn.Sequential( 
                    nn.Conv2d(i, o, kernel_size=(hkernel, vkernel), stride=(hstride, vstride), padding=(hpadding, vpadding)),
                    nn.ReLU(),
                    nn.MaxPool2d(kernel_size=(hpool_size, vpool_size), stride=(hpool_stride, vpool_stride)))
            n = self._add_layer(layer)
            self.layers.append(layer)
            hpixels = compute_size(hpixels, hkernel, hstride, hpadding, f'horizontal conv{n}')
            vpixels = compute_size(vpixels, vkernel, vstride, vpadding, f'vertical conv{n}')
            hpixels = compute_size(hpixels, hpool_size, hpool_stride, 0, f'horizontal pool{n}')
            vpixels = compute_size(vpixels, vpool_size, vpool_stride, 0, f'vertical pool{n}')
        nodes = o * hpixels * vpixels
        if type(linear) == int:
            linear = [ linear ]
        for i in linear:
            layer = nn.Sequential( nn.Linear(nodes, i), nn.ReLU() )
            self._add_layer( layer )
            self.linears.append( layer )
            nodes = i
        
        layer = nn.Linear(nodes, num_classes)
        self._add_layer( layer )
        self.linears.append( layer )
        if final_activation is None:
            self.final_activation = (lambda x:x) if num_classes > 1 else nn.Sigmoid()
        else:
            self.final_activation = final_activation
        if post_forward is None:
            if num_classes == 1:
                self.post_forward = lambda y:torch.round(y)
            else:
                self.post_forward = lambda y:torch.argmax(y, axis=1)
        else:
            self.post_forward = post_forward

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
        return self.final_activation(x)
