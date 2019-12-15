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


class PartialLayer:

    def __init__(self, layer_func, *args, out_shape=None, **kwargs):
        self.layer_func = layer_func
        self.args = args
        self.kwargs = kwargs
        self.out_shape = None
        if not out_shape:
            self.out_shape = args[1]  # TODO: should this be 0?

    def __call__(self, *args, **kwargs):
        self.layer_func(*args, **kwargs)


# applies the shape argument
def _apply_shape(layer_func, shape):
    args = list(layer_func.args)
    kwargs = layer_func.kwargs
    if len(args) > 0:
        args[0] = shape
    return layer_func.layer_func(*args, **kwargs)

# TODO: remove duplicate funcs if possible (want to still be able to run wo loader)

# NOTE: since there is no way to know what is in nested submodules, those will have to handle their own shape inference
# infer the input channels/features from the previous layer for partial layers
# TODO: generalize for things like flatten? pass in loader and get rid of Input? or have option for both?
# TODO: using the loader is the only way to get inference for dense/flatten, since it needs to know the size of the input
def infer_shapes(layers):
    def _infer(layers, current_shape):
        if len(layers) == 0:
            return []
        new_layer = layers[0]
        new_shape = current_shape
        if type(new_layer) is PartialLayer:
            new_shape = new_layer.out_shape
            new_layer = _apply_shape(new_layer, current_shape)
        if len(layers) == 1:
            return [new_layer,]
        new_layers = [new_layer]
        new_layers.extend(_infer(layers[1:], new_shape))
        return new_layers
    assert type(layers[0]) is Input
    return _infer(layers[1:], layers[0].size)


# TODO: device?
# TODO: rename to indicate the data needs to be of same dims each time
def infer_shapes_on_data(layers, loader):
    def _infer(layers, current_shape, partial_model):
        if len(layers) == 0:
            return []
        new_layer = layers[0]
        new_shape = current_shape
        if type(new_layer) is PartialLayer:
            new_shape = new_layer.out_shape
            new_shape = _infer_shape(new_layer.layer_func, new_shape, partial_model, loader)
            new_layer = _apply_shape(new_layer, current_shape)
        if len(layers) == 1:
            return [new_layer, ]
        partial_model.append(new_layer)
        new_layers = [new_layer]
        new_layers.extend(_infer(layers[1:], new_shape, partial_model))
        return new_layers
    assert type(layers[0]) is Input
    return _infer(layers[1:], layers[0].size, [])


# TODO: device?
def _infer_shape(layer, prev_output_shape, partial_model, loader):
    if layer.__name__ is 'Flatten':
        model = nn.Sequential(*partial_model, nn.Flatten())  # TODO: replace with layer and make sure params are passed

        def _run(net, trainer, device):
            def _r(inputs, gtruth):
                return model(inputs)
            return _r
        output = dry_run(model, loader, None, _run)()
        return output.size()[-1]
    else:
        return prev_output_shape
