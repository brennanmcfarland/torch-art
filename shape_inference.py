import torch.nn as nn
from app import dry_run


class Input(nn.Module):
    def __init__(self, input_size=None):
        super(Input, self).__init__()
        self.size = input_size

    def forward(self, x):
        return x


# can be passed a loader, in which case all supported shapes can utilize shape inference, or not, in which case
# only certain layers can utilize shape inference
# custom_layer_types and custom_layer_superclasses are dicts with key types of layer name (string) and layer superclass,
# respectively, and both have values of type lambda prev_layer, prev_layers => shape
# they are for overriding/providing custom shape inference implementation on a layer-by-layer basis
# custom layers take precedence, then custom layer superclasses, then built-in implementation
# device isn't used since it's probably less efficient to move the incrementally created network each time, anyway
# if the loader is used, it's assumed every example will have the same number of channels/features
class ShapeInferer:
    def __init__(self, loader=None, custom_layer_types=None, custom_layer_superclasses=None):
        self.shape = None
        self.loader = loader

        self.custom_layer_types = None
        if custom_layer_types is None:
            self.custom_layer_types = {}

        self.custom_layer_superclasses = None
        if custom_layer_superclasses is None:
            self.custom_layer_superclasses = {}

    def __call__(self, prev_layers):
        return self.infer(prev_layers)

    def infer(self, prev_layers):
        self.shape = self._infer(prev_layers[-1], prev_layers)
        return self.shape

    # helper function
    # feed an example through the data loader for manually capturing the output shape, returns the output tensor
    def _run_prev_layers(self, prev_layers, layer_type):
        model = nn.Sequential(*prev_layers)

        def _run(net, trainer, device):
            def _r(inputs, gtruth):
                return model(inputs)
            return _r

        if self.loader is not None:
            return dry_run(model, self.loader, None, _run)()
        else:
            raise TypeError("A data loader must be provided for shape inference with " + layer_type.__name__)

    def _infer(self, prev_layer, prev_layers):
        layer_type = type(prev_layer)
        if layer_type.__name__ in self.custom_layer_types:
            # the value returned from the custom layer inference
            return self.custom_layer_types[layer_type.__name__](prev_layer, prev_layers)
        elif any((issubclass(layer_type, c) for c in self.custom_layer_superclasses.keys())):
            # the value returned from the custom layer superclass inference
            return self.custom_layer_superclasses[layer_type.__name__](prev_layer, prev_layers)
        elif layer_type.__name__ is 'Input':
            # the Input's size if it has one, else get an example from the data loader
            if prev_layer.size is not None:
                return prev_layer.size
            else:
                inputs, _ = iter(self.loader).next()
                return inputs.size()[1]
        elif issubclass(layer_type, nn.modules.conv._ConvNd) or layer_type.__name__ is 'Linear':
            # the out_features dim
            return prev_layer.weight.size()[0]
        elif layer_type.__name__ is 'Flatten':
            # feed an example through the data loader and manually capture the output shape
            return self._run_prev_layers(prev_layers, layer_type).size()[-1]
        elif (layer_type.__name__ is 'ReLU'
              or issubclass(layer_type, nn.modules.pooling._MaxPoolNd)):
            # reuse and pass along the previously inferred shape unchanged
            return self._infer(prev_layers[-2], prev_layers[:-1])
        else:
            raise NotImplementedError("No shape inference implementation for layer of type " + layer_type.__name__)
