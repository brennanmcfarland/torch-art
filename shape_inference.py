import torch.nn as nn
import functools
from app import dry_run

# TODO: this whole area is a mess


# TODO: not sure if this should actually go here
class Input(nn.Module):
    def __init__(self, input_size):
        super(Input, self).__init__()
        self.size = input_size

    def forward(self, x):
        return x


# TODO: note or make it not just work with statically sized input, maybe not with input at all if possible in some cases
class ShapeInferer:
    def __init__(self, loader):
        self.shape = None
        self.loader = loader

    def infer(self, layer, prev_layers):
        self.shape = self._infer(layer, prev_layers[-1], prev_layers)
        return self.shape

    # TODO: generalize
    def _infer(self, layer, prev_layer, prev_layers):
        if type(prev_layer).__name__ is 'Input':
            return prev_layer.size
        elif type(prev_layer).__name__ is 'Conv2d' or type(prev_layer).__name__ is 'Linear':
            return prev_layer.weight.size()[0]  # the out_features dim
        elif type(prev_layer).__name__ is 'Flatten':
            model = nn.Sequential(*prev_layers,
                                  nn.Flatten())  # TODO: replace with layer and make sure params are passed

            def _run(net, trainer, device):
                def _r(inputs, gtruth):
                    return model(inputs)

                return _r

            output = dry_run(model, self.loader, None, _run)()
            return output.size()[-1]
        else:
            return self._infer(prev_layer, prev_layers[-2], prev_layers[:-1])
