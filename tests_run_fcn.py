from __future__ import division
import os
import numpy as np
from tensorflow.contrib.keras.python.keras.datasets import mnist
from tensorflow.contrib.keras.python.keras.models import Sequential, Model
from tensorflow.contrib.keras.python.keras.layers import Dense, Flatten
from tensorflow.contrib.keras.python.keras.layers import Conv2D
from tensorflow.contrib.keras.python.keras.layers.normalization import BatchNormalization
from scipy.ndimage import imread
from tensorflow.contrib.keras.python.keras import backend
from _dc_custom_layers import dilated_MaxPool2D


x = imread('tests/nuc0/nuclear.png')
print x.shape
model = Sequential([
    # 61 x 61
    Conv2D(64, dilation_rate=1, kernel_size=(3, 3), activation='relu', input_shape=(1080, 1280, 1)),
    BatchNormalization(),
    # 64 x 61 x 61
    Conv2D(64, kernel_size=(4, 4), activation='relu'),
    dilated_MaxPool2D(dilation_rate=1, pool_size=(2, 2)),
    # 64 x 30 x 30
    Conv2D(64, dilation_rate=2, kernel_size=(3, 3), activation='relu', padding='valid'),
    Conv2D(64, dilation_rate=2, kernel_size=(3, 3), activation='relu', padding='valid'),
    dilated_MaxPool2D(dilation_rate=2, pool_size=(2, 2)),
    Conv2D(64, dilation_rate=4, kernel_size=(3, 3), activation='relu'),
    Conv2D(64, dilation_rate=4, kernel_size=(3, 3), activation='relu'),
    dilated_MaxPool2D(dilation_rate=4, pool_size=(2, 2)),
    Conv2D(200, dilation_rate=1, kernel_size=(4, 4), activation='relu'),
    Conv2D(200, dilation_rate=1, kernel_size=(1, 1), activation='relu'),
    Conv2D(3, kernel_size=(1, 1), activation='softmax'),
])

model.load_weights('weights.03-0.10-0.96-0.10-0.96.hdf5')
# model.load_weights('tests1.hdf5')
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
