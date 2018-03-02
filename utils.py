
import os
import imp
from tensorflow.contrib.keras.python.keras.engine.topology import Layer
from tensorflow.contrib.keras.python.keras import backend
from tensorflow.contrib.keras.python.keras.layers import Conv2D, MaxPooling2D
from _dc_custom_layers import dilated_MaxPool2D
from tensorflow.contrib.keras.python.keras.models import Sequential


class Squeeze(Layer):
    def __init__(self, output_dim, **kwargs):
        self.output_dim = output_dim
        super(Squeeze, self).__init__(**kwargs)

    def call(self, x):
        x = backend.squeeze(x, axis=2)
        return backend.squeeze(x, axis=1)

    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[3])


def convert_model_patch2full(model, shape):
    dr = 1
    new_model = Sequential()
    for nl, layer in enumerate(model.layers):
        if isinstance(layer, Squeeze):
            continue
        if isinstance(layer, MaxPooling2D):
            newl = dilated_MaxPool2D(dilation_rate=dr)
            newl = newl.from_config(layer.get_config())
            newl.strides, newl.dilation_rate = (1, 1), dr
            new_model.add(newl)
            dr = dr * 2
            continue
        if isinstance(layer, Conv2D):
            if not layer.kernel_size == (1, 1):
                if nl == 0:
                    newl = Conv2D(layer.filters, layer.kernel_size, input_shape=shape)
                    newl = newl.from_config(layer.get_config())
                    newl.dilation_rate = (dr, dr)
                    new_model.add(newl)
                if not nl == 0:
                    layer.dilation_rate = (dr, dr)
                    new_model.add(layer)
            else:
                newl = Conv2D(layer.filters, layer.kernel_size, input_shape=layer.input_shape[1:])
                new_model.add(newl.from_config(layer.get_config()))
        else:
            new_model.add(layer)
    return new_model


def load_model_py(path):
    fname = os.path.basename(path).split('.')[0]
    module = imp.load_source('tests_model', path)
    return module.model