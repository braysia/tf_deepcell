from __future__ import division
import os
import numpy as np
from tensorflow.contrib.keras.python.keras.datasets import mnist
from tensorflow.contrib.keras.python.keras.models import Sequential
from tensorflow.contrib.keras.python.keras.layers import Conv2D, MaxPooling2D
from tensorflow.contrib.keras.python.keras.layers.normalization import BatchNormalization
from scipy.ndimage import imread
from tensorflow.contrib.keras.python.keras import backend
from _dc_custom_layers import dilated_MaxPool2D
from utils import convert_model_patch2full, load_model_py
import importlib

x = imread('tests/nuc1/nuclear.png')
print x.shape
model_file = 'models/tests_model.py'
model = load_model_py(model_file)
# f = importlib.import_module(model_file)
shape = x.shape

model = convert_model_patch2full(model, shape)
model.load_weights('weights.03-0.10-0.96-0.10-0.96.hdf5')

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
