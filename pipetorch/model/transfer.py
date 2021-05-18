import torch
import torch.nn as nn
import torchvision.models as models
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
    def __init__(self):
        super().__init__()

    def set_last_linear(self, out_features):
        container = self.last_container()
        name, last = container._modules.popitem()
        container.add_module(name, nn.Linear(last.in_features, out_features))

    def last_container(self):
        return last_container(self)

    def freeze(self):
        for c in list(self.children())[:-1]:
            for p in c.parameters():
                p.requires_grad=False

    def unfreeze(self):
        for c in list(self.children())[:-1]:
            for p in c.parameters():
                p.requires_grad=True

class Transfer(DLModel):
    def __init__(self, output_nodes=None, model=models.resnet34, pretrained=True):
        super().__init__()
        os.environ['TORCH_HOME'] = '/datb/torch'
        self.model = model(pretrained=pretrained)
        if output_nodes is not None:
            self.set_last_linear(output_nodes)

    def children(self):
        return self.model.children()

    def last_container(self):
        return last_container(self.model)

    def forward(self, x):
        return self.model( x )

    def post_forward(self, y):
        return torch.argmax(y, axis=1)
