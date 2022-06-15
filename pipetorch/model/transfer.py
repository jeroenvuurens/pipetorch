import torch
import torch.nn as nn
import torchvision.models as models
from torch.hub import get_dir
import os

def last_container(last):
    try:
        l = last_container(last.children())
        if l is not None:
            return l
    except: pass
    try:
        if len(last._modules) > 0 and next(reversed(last._modules.values())).out_features > 0:
            return last
    except: pass
    
class DLModel(nn.Module):
    """
    A general purpose Deep model that will split an existing model between a base (lower layers) 
    and a final layer. This class provides methods to:
    - split_final: split off the final layer from a model
    - replace_final: replace the final layer with a Linear and a given number of output_nodes
    - freeze: freeze the base so that the weights are not trained
    - unfreeze: unfreeze the base, so the weights are trained
    
    Typical use for this class is for Transfer Learning.
    """
    def __init__(self, model):
        super().__init__()
        self.base, self.final = self.split_final(model)

    def split_final(self, model):
        container = last_container(model)
        name, last = container._modules.popitem()
        container.add_module(name, nn.Sequential())
        return model, last
    
    def replace_final(self, out_features):
        self.final = nn.Linear(self.final.in_features, out_features)

    def freeze(self):
        for c in list(self.base.children()):
            for p in c.parameters():
                p.requires_grad=False

    def unfreeze(self):
        for c in list(self.base.children()):
            for p in c.parameters():
                p.requires_grad=True

class Transfer(DLModel):
    """
    Create a model based on a (pretrained) torchvision model. The last layer is cut from the model
    and replaced with a linear that maps towards the number of given output_nodes. This model
    supports two-phase transfer learning through the freeze() and unfreeze() methods that
    allow to (un)freeze the model except the last layer.
    
    Args:
        out_features: int (None)
            if not None, the last layer in the model is replaced by a linear layer that maps
            to the given number of output nodes.
            
        model: tochvision.models.* (resnet34)
            the model to transfer. This class was only tested with models from torchvision.models
            but possibly works with other models as well.
            
        pretrained: bool (True)
            is passed to the torchvision model function, when True the model is downloaded from
            a PyTorch repository. The torchvision models are usually trained on ImageNet.
            
        user: bool (False)
            the downloaded model is cached in an indicated location. By default, a shared folder
            ~/.pipetorch to save storage on a server with multiple users.
            Alternatively, when user=True the model is stored in a user specific ~/.pipetorchuser 
            folder. In case the users are not allowed write access on the shared folder.
    """
    
    def __init__(self, out_features=None, model=models.resnet34, pretrained=True, user=False, output_nodes=None):
        os.environ['TORCH_HOME'] = '~/.pipetorchuser' if user else '~/.pipetorch'
        super().__init__(model(pretrained=pretrained))
        out_features = out_features or output_nodes
        if out_features is not None:
            self.replace_final(out_features)

    def forward(self, *X):
        h = self.base( *X )
        return self.final( h )
