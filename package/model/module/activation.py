import torch
import torch.nn as nn
import torch.nn.functional as F

class Mish(nn.Module):
    def __init__(self):
        super().__init__()
        #print("Mish activation loaded...")
    def forward(self,x):
        x = x * (torch.tanh(F.softplus(x)))
        return x

activations = {'ReLU': nn.ReLU,
               'LeakyReLU': nn.LeakyReLU,
               'ReLU6': nn.ReLU6,
               'SELU': nn.SELU,
               'ELU': nn.ELU,
               'GELU': nn.GELU,
               'SiLU':nn.SiLU,
               'Mish':Mish(),
               None: nn.Identity
               }


def act_layers(name):
    assert name in activations.keys()
    if name == 'LeakyReLU':
        return nn.LeakyReLU(negative_slope=0.1, inplace=True)
    elif name == 'GELU':
        return nn.GELU()
    elif name == 'SiLU':
        return nn.SiLU()
    elif name == 'Mish':
        return Mish()

    else:
        return activations[name](inplace=True)
