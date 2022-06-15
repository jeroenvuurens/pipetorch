#from torchvision.models import *
import torch
import torch.nn as nn

identity=lambda x:x

# class SingleLayerPerceptron(nn.Module):
#     def __init__(self, input, output, last_activation=identity):
#         super().__init__()
#         self.w1 = nn.Linear(input, output)
#         self.a1 = last_activation

#     def forward(self, x):
#         return self.a1(self.w1(x))
#         #return pred_y.view(-1)
    
# class SingleLayerPerceptron_BinaryClass(SingleLayerPerceptron):
#     def __init__(self, input, output):
#         super().__init__(input, output, nn.Sigmoid())

#     def post_forward(self, y):
#         return (y > 0.5).float()

# class SingleLayerPerceptron_MultiClass(SingleLayerPerceptron):
#     def __init__(self, input, output):
#         super().__init__(input, output, nn.LogSoftmax(dim=1))

def flatten_r_image(x):
        return  x[:,0,:,:].view(x.shape[0], -1)

class Perceptron(nn.Module):
    """
    Class that implements a generic MultiLayerPerceptron
    
    Args:
        *with: int
            Sequence of at least two ints, that provide the widths for all the layers in the network.
            The first width should match the number of input features, the last width should match the
            numbetr of target variables (usually 1).
            
        preprocess: func (identity)
            function that will be used on the input first. The default means no preprocessing
            
        inner_activation: func (nn.ReLU())
            activation function that is used on all layers except the output
            
        drop_prob: float (None)
            if provided, Dropout layers are added in between all layers but the last, with a fixed
            dropout probability.
            
        last_activation: func (None)
            the activation function used on the last layer. The most common choices are None for regression
            nn.Sigmoid() for binary classification and nn.Softmax() for multi-label classification.
    """
    
    def __init__(self, *width, preprocess=identity, inner_activation=nn.ReLU(), drop_prob=None, last_activation=None):
        super().__init__()
        self.actions = [preprocess]
        for n, (i, o) in enumerate(zip(width[:-1], width[1:])):
            l = nn.Linear(i, o)
            self.actions.append(l)
            self.__setattr__(f'w{n+1}', l)
            if n < len(width) - 2:
                if drop_prob is not None:
                    self.actions.append(nn.Dropout(p=drop_prob))
                    self.__setattr__(f'drop{n+1}', self.actions[-1])
                self.actions.append(inner_activation)
                self.__setattr__(f'activation{n+1}', self.actions[-1])
            elif last_activation is not None:
                self.actions.append(last_activation)
                self.__setattr__(f'activation{n+1}', self.actions[-1])
        #if width[-1] == 1:
        #    self.reshape = (-1)
        #else:
        #    self.reshape = (-1, width[-1])
        
    def forward(self, x):
        for a in self.actions:
            x = a(x)
        return x #.view(self.reshape)

# class MultiLayerPerceptron(Perceptron):
#     pass
    
# class MultiLayerPerceptron_BinaryClass(MultiLayerPerceptron):
#     def __init__(self, *width, preprocess=identity, inner_activation=nn.ReLU(), drop_prob=None):
#         super().__init__(*width, preprocess=preprocess, inner_activation=inner_activation, drop_prob=drop_prob, last_activation=nn.Sigmoid())

#     def post_forward(self, y):
#         return (y > 0.5).float()

# class MultiLayerPerceptron_MultiClass(MultiLayerPerceptron):
#     def __init__(self, *width, preprocess=identity, inner_activation=nn.ReLU(), drop_prob=None):
#         super().__init__(*width, preprocess=preprocess, inner_activation=inner_activation, drop_prob=drop_prob)
        
#     def post_forward(self, y):
#         return torch.argmax(y, axis=1)
        
# class TwoLayerPerceptron(nn.Module):
#     def __init__(self, input, hidden, output, last_activation=None):
#         super().__init__()
#         self.w1 = nn.Linear(input, hidden)
#         self.a1 = nn.ReLU()
#         self.w2 = nn.Linear(hidden, output)
#         if last_activation:
#             self.a2 = last_activation

#     def forward(self, x):
#         x = self.a1(self.w1(x))
#         pred_y = self.a2(self.w2(x))
#         return pred_y #.view(-1)

#     def post_forward(self, y):
#         return y 

# class TwoLayerPerceptron_BinaryClass(TwoLayerPerceptron):
#     def __init__(self, input, hidden, output):
#         super().__init__(input, hidden, output, last_activation=nn.Sigmoid())

#     def post_forward(self, y):
#         return (y > 0.5).float()

# class TwoLayerPerceptron_MultiClass(TwoLayerPerceptron):
#     def __init__(self, input, hidden, output):
#         super().__init__(input, hidden, output, last_activation=nn.LogSoftmax(dim=1))

# def zero_embedding(rows, columns):
#     e = nn.Embedding(rows, columns)
#     e.weight.data.zero_()
#     return e

# class factorization(nn.Module):
#     def __init__(self, n_users, n_items, n_factors=20):
#         super().__init__()
#         self.user_factors = nn.Embedding( n_users,n_factors)
#         self.item_factors = nn.Embedding( n_items,n_factors)
#         self.user_bias = zero_embedding( n_users, 1)
#         self.item_bias = zero_embedding( n_items, 1)
#         self.fc = nn.Linear(n_factors, 4)
        
#     def forward(self, X):
#         user = X[:,0] - 1
#         item = X[:,1] - 1
#         return (self.user_factors(user) * self.item_factors(item)).sum(1) + self.user_bias(user).squeeze() + self.item_bias(item).squeeze()

