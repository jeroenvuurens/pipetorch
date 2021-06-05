from torch.utils.data import Dataset
from torch import Tensor
import torch
import numpy as np

def sequence_tensors(X, y, window):
    X = SequenceTensor(X, window)
    y = SequenceTargetTensor(y)
    indices = torch.tensor(np.intersect1d(X._indices, y._indices))
    X._indices = indices
    y._indices = indices
    X._length = len(indices)
    y._length = len(indices)
    return X, y, indices 

def tensors(X, y):
    X = PTTensor(X)
    y = PTTensor(y)
    indices = torch.tensor(np.intersect1d(X._indices, y._indices))
    X._indices = indices
    y._indices = indices
    X._length = len(indices)
    y._length = len(indices)
    return X, y, indices   

class PTTensor(Tensor):
    @staticmethod
    def __new__(cls, array, *args, **kwargs):
        return super().__new__(cls, array, *args, **kwargs)
    
    def __init__(self, array, *args, **kwargs):
        a = torch.isnan(self[:])
        while len(a.shape) > 1:
            a = torch.any(a, -1)
        self._indices = torch.where(~a)[0]
        
    def __getitem__(self, index):
        if isinstance( index, slice ) :
            if len(self) == 0:
                return torch.zeros(self.size())
            indices = list(range(*index.indices(len(self))))
            try:
                indices = self._indices[indices]
            except:pass
            return super().__getitem__(indices)
        try:
            return super().__getitem__(self._indices[index])
        except:
            return super().__getitem__(index)
           
    def __len__(self):
        try:
            return len(self._indices)
        except:
            return super().__len__()
    
    def size(self, dim=None):
        if dim == 0:
            return len(self)
        if dim is not None and dim > 0:
            return super().size(dim)
        return torch.Size([len(self)] + list(super().size())[1:])

class SequenceTensor:
#    @staticmethod
#    def __new__(cls, array, window, *args, **kwargs):
#        return super().__new__(cls, array, *args, **kwargs)
    
    def __init__(self, array, window):
        self.tensor = array if isinstance(array,Tensor) else torch.tensor(array)
        self.window = window
        
    @property
    def window(self):
        return self._window
    
    @window.setter
    def window(self, window):
        self._window = window
        self._length = max(0, len(self.tensor) - window + 1)
        self._indices = list(range(self._length))
        if self._length > 0:
            a = torch.isnan(self[:])
            while len(a.shape) > 1:
                a = torch.any(a, -1)
            self._indices = torch.where(~a)[0]
            self._length = len(self._indices)
    
    def __getitem__(self, index):
        if isinstance( index, slice ) :
            if len(self) == 0:
                return torch.zeros(self.size())
            return torch.cat([self[ii].unsqueeze(0) for ii in range(*index.indices(self._length))], axis=0)
        return self.tensor[self._indices[index]:self._indices[index]+self.window]
    
    def __len__(self):
        return self._length
    
    def size(self, dim=None):
        if dim == 0:
            return len(self)
        if dim == 1:
            return self.window
        if dim is not None and dim > 0:
            return self.tensor.size(dim - 1)
        return torch.Size([len(self), self.window] + list(self.tensor.size())[1:])

    def clone(self, *args, **kwargs): 
        return SequenceTensor(super().clone(*args, **kwargs), self.window)
    
    def to(self, *args, **kwargs):
        new_obj = SequenceTensor(self.tensor.to(*args, **kwargs), window)
        new_obj.window = self.window
        new_obj.tensor.requires_grad=self.tensor.requires_grad
        return new_obj
    
class SequenceTargetTensor:
#    @staticmethod
#    def __new__(cls, array, *args, **kwargs):
#        return super().__new__(cls, array, *args, **kwargs)

    def __init__(self, array):
        self.tensor = array if isinstance(array,Tensor) else torch.tensor(array)
        if len(array.shape) > 0:
            self._indices = list(np.where(~np.isnan(array))[0])
        else:
            self._indices = []
        
    def __getitem__(self, index):
        if isinstance( index, slice ) :
            if len(self) == 0:
                return torch.zeros(0)
            indices = list(range(*index.indices(len(self))))
            try:
                indices = self._indices[indices]
            except:pass
            return self.tensor[indices]
        return self.tensor[self._indices[index]]

    def __len__(self):
        return len(self._indices)
    
    def size(self, dim=None):
        if dim == 0:
            return len(self)
        if dim is not None and dim > 0:
            return self.tensor.size(dim)
        return torch.Size([len(self)])

    def reshape(self, *view):
        return self[:].reshape(*view)
    
    @property
    def shape(self):
        return self.size()
    