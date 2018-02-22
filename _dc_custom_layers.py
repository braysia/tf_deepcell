"""
dc_custom_layers.py

Custom layers for convolutional neural networks

@author: David Van Valen
"""

"""
Import python packages
"""

import numpy as np
from numpy import array
import matplotlib
import matplotlib.pyplot as plt
import shelve
from contextlib import closing

import os
import glob
import re
import numpy as np
import fnmatch
import tifffile as tiff
from numpy.fft import fft2, ifft2, fftshift
from skimage.io import imread
from scipy import ndimage
import threading
import scipy.ndimage as ndi
from scipy import linalg
import re
import random
import itertools
import h5py
import datetime

from skimage.measure import label, regionprops
from skimage.segmentation import clear_border
from scipy.ndimage.morphology import binary_fill_holes
from skimage import morphology as morph
from numpy.fft import fft2, ifft2, fftshift
from skimage.io import imread
from skimage.filters import threshold_otsu
import skimage as sk
from sklearn.utils.linear_assignment_ import linear_assignment
from sklearn.utils import class_weight

import tensorflow as tf
from tensorflow.contrib.keras.python.keras import backend as K
from tensorflow.contrib.keras.python.keras.layers import Layer, InputSpec, Input, Activation, Dense, Flatten, BatchNormalization, Conv2D, MaxPool2D, AvgPool2D, Concatenate
from tensorflow.contrib.keras.python.keras.preprocessing.image import random_rotation, random_shift, random_shear, random_zoom, random_channel_shift, apply_transform, flip_axis, array_to_img, img_to_array, load_img, ImageDataGenerator, Iterator, NumpyArrayIterator, DirectoryIterator
from tensorflow.contrib.keras.python.keras.callbacks import ModelCheckpoint, LearningRateScheduler
from tensorflow.contrib.keras.python.keras import activations, initializers, losses, regularizers, constraints
from tensorflow.contrib.keras.python.keras.utils import conv_utils

# from dc_helper_functions import *

"""
Custom layers
"""
class dilated_MaxPool2D(Layer):
	def __init__(self, pool_size=(2, 2), strides=None, dilation_rate = 1, padding='valid',
				data_format=None, **kwargs):
		super(dilated_MaxPool2D, self).__init__(**kwargs)
		data_format = conv_utils.normalize_data_format(data_format)
		if dilation_rate != 1:
			strides = (1,1)
		elif strides is None:
			strides = (1,1)
		self.pool_size = conv_utils.normalize_tuple(pool_size, 2, 'pool_size')
		self.strides = conv_utils.normalize_tuple(strides, 2, 'strides')
		self.dilation_rate = dilation_rate
		self.padding = conv_utils.normalize_padding(padding)
		self.data_format = conv_utils.normalize_data_format(data_format)
		self.input_spec = InputSpec(ndim=4)

	def compute_output_shape(self, input_shape):
		if self.data_format == 'channels_first':
			rows = input_shape[2]
			cols = input_shape[3]
		elif self.data_format == 'channels_last':
			rows = input_shape[1]
			cols = input_shape[2]

		rows = conv_utils.conv_output_length(rows, pool_size[0], padding = self.padding, stride = self.strides[0], dilation = dilation_rate)
		cols = conv_utils.conv_output_length(cols, pool_size[1], padding = self.padding, stride = self.strides[1], dilation = dilation_rate)
		
		if self.data_format == 'channels_first':
			return (input_shape[0], input_shape[1], rows, cols)
		elif self.data_format == 'channels_last':
			return (input_shape[0], rows, cols, input_shape[3])

	def  _pooling_function(self, inputs, pool_size, dilation_rate, strides, padding, data_format='channels_first'):
		backend = K.backend()
		
		#dilated pooling for tensorflow backend
		if backend == "theano":
			Exception('This version of DeepCell only works with the tensorflow backend')

		if data_format == 'channels_first':
			df = 'NCHW'
		elif data_format == 'channel_last':
			df = 'NHWC'
		df = 'NHWC'

		if self.padding == "valid":
			padding_input = "VALID"

		if self.padding == "same":
			padding_input = "SAME"
			
		output = tf.nn.pool(inputs, window_shape = pool_size, pooling_type = "MAX", padding = padding_input,
							dilation_rate = (dilation_rate, dilation_rate), strides = strides, data_format = df)

		return output

	def call(self, inputs):
		output = self._pooling_function(inputs = inputs, 
										pool_size=self.pool_size,
										strides = self.strides,
										dilation_rate = self.dilation_rate,
										padding = self.padding,
										data_format = self.data_format)
		return output

	def get_config(self):
		config = {'pool_size': self.pool_size,
					'padding': self.padding,
					'dilation_rate': self.dilation_rate,
					'strides': self.strides,
					'data_format': self.data_format}
		base_config = super(dilated_MaxPool2D, self).get_config()
		return dict(list(base_config.items()) + list(config.items()))
