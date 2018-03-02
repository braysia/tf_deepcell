from __future__ import division
import os
import numpy as np
from tensorflow.contrib.keras.python.keras.datasets import mnist
from tensorflow.contrib.keras.python.keras.models import Sequential, Model
from tensorflow.contrib.keras.python.keras.layers import Dense, Flatten
from tensorflow.contrib.keras.python.keras.layers import Conv2D, MaxPooling2D
from tensorflow.contrib.keras.python.keras.layers.normalization import BatchNormalization
from scipy.ndimage import imread
from tensorflow.contrib.keras.python.keras import backend
from _dc_custom_layers import dilated_MaxPool2D


# from tensorflow.contrib.keras.python.keras.engine.topology import Layer
# from tensorflow.contrib.keras.python.keras import backend
# class Squeeze(Layer):
#     def __init__(self, output_dim, **kwargs):
#         self.output_dim = output_dim
#         super(Squeeze, self).__init__(**kwargs)

#     def call(self, x):
#         x = backend.squeeze(x, axis=2)
#         return backend.squeeze(x, axis=1)

#     def compute_output_shape(self, input_shape):
#         return (input_shape[0], input_shape[3])

# def convert_model_patch2full(model):
#     dr = 1
#     new_model = Sequential()
#     for nl, layer in enumerate(model.layers):
#         print nl
#         if isinstance(layer, Squeeze):
#             continue
#         if isinstance(layer, MaxPooling2D):
#             newl = dilated_MaxPool2D(dilation_rate=dr)
#             newl.from_config(layer.get_config())
#             new_model.add(newl)
#             dr = dr * 2
#             continue
#         if isinstance(layer, Conv2D):
#             if not layer.kernel_size == (1, 1):
#                 layer.dilation_rate = dr
#                 newl = Conv2D(layer.filters, layer.kernel_size, dilation_rate=dr, input_shape=layer.input_shape[1:])
#                 newl.from_config(layer.get_config())
#                 new_model.add(newl)
#             else:
#                 newl = Conv2D(layer.filters, layer.kernel_size, input_shape=layer.input_shape[1:])
#                 newl.from_config(layer.get_config())
#                 new_model.add(newl)
#         else:
#             new_model.add(layer)
#     return new_model

    # for nl, layer in enumerate(model.layers):
    #     if isinstance(layer, MaxPooling2D):
    #         layer._set_dilation(dr)
    #         # layer.dilation_rate = dr
    #         dr = dr * 2
    #     if isinstance(layer, Conv2D):
    #         if not layer.kernel_size == (1, 1):
    #             layer.dilation_rate = dr
    #     if isinstance(layer, Squeeze):
    #         model.layers.pop(nl)
    # return model

x = imread('tests/nuc1/nuclear.png')
print x.shape
model = Sequential([
    # 61 x 61
    Conv2D(64, dilation_rate=1, kernel_size=(3, 3), activation='relu', input_shape=(1080, 1280, 1)),
    BatchNormalization(),
    # 64 x 61 x 61
    Conv2D(64, kernel_size=(4, 4), activation='relu'),
    # dilated_MaxPool2D(dilation_rate=1, pool_size=(2, 2)),
    dilated_MaxPool2D(dilation_rate=1, pool_size=(2, 2)),
    # 64 x 30 x 30
    Conv2D(64, dilation_rate=2, kernel_size=(3, 3), activation='relu', padding='valid'),
    Conv2D(64, dilation_rate=2, kernel_size=(3, 3), activation='relu', padding='valid'),
    dilated_MaxPool2D(dilation_rate=2, pool_size=(2, 2)),
    Conv2D(64, dilation_rate=4, kernel_size=(3, 3), activation='relu'),
    Conv2D(64, dilation_rate=4, kernel_size=(3, 3), activation='relu'),
    dilated_MaxPool2D(dilation_rate=4, pool_size=(2, 2)),
    Conv2D(200, dilation_rate=8, kernel_size=(4, 4), activation='relu'),
    Conv2D(200, dilation_rate=1, kernel_size=(1, 1), activation='relu'),
    Conv2D(3, kernel_size=(1, 1), activation='softmax'),
])


# model = Sequential([
#     # 61 x 61
#     Conv2D(64, kernel_size=(3, 3), activation='relu', input_shape=(1080, 1280, 1)),
#     BatchNormalization(),
#     # 64 x 61 x 61
#     Conv2D(64, kernel_size=(4, 4), activation='relu'),
#     MaxPooling2D(),
#     # 64 x 30 x 30
#     Conv2D(64, kernel_size=(3, 3), activation='relu'),
#     Conv2D(64, kernel_size=(3, 3), activation='relu'),
#     MaxPooling2D(),
#     Conv2D(64, kernel_size=(3, 3), activation='relu'),
#     Conv2D(64, kernel_size=(3, 3), activation='relu'),
#     MaxPooling2D(),
#     Conv2D(200, kernel_size=(4, 4), activation='relu'),
#     Conv2D(200, kernel_size=(1, 1), activation='relu'),
#     Conv2D(3, kernel_size=(1, 1), activation='softmax'),
#     # Squeeze(3),
# ])

# model = convert_model_patch2full(model)
model.load_weights('weights.03-0.10-0.96-0.10-0.96.hdf5')
# model.load_weights('weights.11-0.18-0.92-0.13-0.94.hdf5')
# model.load_weights('tests1.hdf5')
# from tensorflow.contrib.keras import optimizers
# opt = optimizers.SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
# model.compile(optimizer=opt, loss='sparse_categorical_crossentropy', metrics=['accuracy'])

model.summary()
# assert False
evaluate_model = backend.function(
    [model.layers[0].input, backend.learning_phase()],
    [model.layers[-1].output]
    )

x = np.expand_dims(x, -1)
x = np.expand_dims(x, 0)
cc = evaluate_model([x, 0])[0]
print cc.shape

import tifffile as tiff
tiff.imsave('test0.tif', cc[0, :, :, 0])
tiff.imsave('test1.tif', cc[0, :, :, 1])
tiff.imsave('test2.tif', cc[0, :, :, 2])
