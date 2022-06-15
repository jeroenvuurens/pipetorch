
from torch.utils.data import Dataset
from inspect import signature
import torch
import numpy as np

class TransformationXY:
    """
    To model a transformation that TransformableDataset will recognize as a transformation on an (x,y) pair
    subclass this class and add it to the transformations.
    """
    def __call__(self, x, y):
        pass

class TransformableDataset(Dataset):
    """
    Subclass of a PyTorch Dataset, that allows optional transformation of the data that is returned.
    TransformableDataset can transform either the input x, output y or both.
    
    Arguments:
        dataset: A Dataset that supports __len__() and __getitem__()
        
        dtype: (None)
            The dtype of the input, if None torch.float32
        
        *transforms: [ callable ]
            Each callable is a function that is called as x = func(x) to transform only x
            or as x, y = func(x, y) to transform both x and y
    """
    
    def __init__(self, dataset, dtype, *transforms):
        self.dataset = dataset
        self._dtype = dtype
        self.transforms = list(transforms)
        try:
            self.tensors = dataset.tensors
        except: pass

    def __getitem__(self, index):
        """
        Arguments:
            index: int
                number in the range 0 - len(dataset) that identifies a data example.
                
        Returns: (tensor, tensor)
            the pair of x, y (input, output) at the given index, that is transformed by the given transformation functions.
        """
        
        x, y = self.dataset[index]
        for t in self.transforms:
            if isinstance(t, TransformationXY):
                x, y = t(x, y)
            else:
                x = t(x)
        self.debugx = x
        if self._dtype is None:
            x = x.type(torch.FloatTensor)
        elif self._dtype:
            if np.issubdtype(self._dtype, np.number):
                x = x.type(self._dtype)
            else:
                x = x.type(torch.FloatTensor)
       
        return x, y
    
    def __len__(self):
        return len(self.dataset)  
