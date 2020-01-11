import torch.nn as nn


# given an iterable of layers, create a network
def from_iterable(layers):
    return nn.Sequential(*layers)
