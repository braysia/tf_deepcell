import numpy as np
from tensorflow.contrib.keras.python.keras.datasets import mnist

from tensorflow.contrib.keras.python.keras.models import Sequential, Model
from tensorflow.contrib.keras.python.keras.layers import Dense, Dropout, Activation, Flatten
from tensorflow.contrib.keras.python.keras.layers import Conv2D, MaxPooling2D, AveragePooling2D
from tensorflow.contrib.keras.python.keras.layers import Input, merge
from tensorflow.contrib.keras.python.keras.layers.normalization import BatchNormalization
from tensorflow.contrib.keras.python.keras.layers.merge import add
from tensorflow.contrib.keras import optimizers
from tensorflow.contrib.keras import callbacks
from tensorflow.contrib.keras import metrics

from tensorflow.contrib.keras import preprocessing
from scipy.ndimage import imread
from sklearn.feature_extraction.image import extract_patches_2d

x_train = imread('tests/nuc0/nuclear.png')
y_train = np.zeros(x_train.shape)
y0 = imread('tests/nuc0/feature_0.png').astype(bool)
y1 = imread('tests/nuc0/feature_1.png').astype(bool)
y_train[y0] = 1
y_train[y1] = 2

img_stack = np.dstack((x_train, y_train))

ph, pw = 61, 61
nsamples = 100000
frac_test = 0.1

patch_stack = extract_patches_2d(img_stack, patch_size=[ph, pw], max_patches=nsamples)
x_train = patch_stack[:-int(nsamples * frac_test), :, :, 0].astype(np.float32)
x_test = patch_stack[-int(nsamples * frac_test):, :, :, 0].astype(np.float32)
y_train = patch_stack[:-int(nsamples * frac_test), int(np.floor(ph/2)), int(np.floor(pw/2)), -1].astype(np.int8)
y_test = patch_stack[-int(nsamples * frac_test):, int(np.floor(ph/2)), int(np.floor(pw/2)), -1].astype(np.int8)
x_train = np.expand_dims(x_train, -1)
x_test = np.expand_dims(x_test, -1)

model = Sequential([
    Conv2D(64, kernel_size=(3, 3), activation='relu', input_shape=(61, 61, 1)),
    BatchNormalization(),
    Conv2D(64, kernel_size=(4, 4), activation='relu'),
    MaxPooling2D(),
    Conv2D(64, kernel_size=(3, 3), activation='relu'),
    Conv2D(64, kernel_size=(3, 3), activation='relu'),
    MaxPooling2D(),
    Conv2D(64, kernel_size=(3, 3), activation='relu'),
    Conv2D(64, kernel_size=(3, 3), activation='relu'),
    MaxPooling2D(),
    Conv2D(200, kernel_size=(4, 4), activation='relu'),
    Flatten(),
    Dense(200, activation='relu'),
    Dense(3, activation='softmax'),
])

model.summary()

# opt = optimizers.Adam()
opt = optimizers.SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(optimizer=opt,
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

csv_logger = callbacks.CSVLogger('training.log')

datagen = preprocessing.image.ImageDataGenerator(rotation_range=90, shear_range=0, 
                                                 horizontal_flip=True, vertical_flip=True)
history = model.fit_generator(datagen.flow(x_train, y_train, batch_size=250),
                              steps_per_epoch=len(y_train),  # FIXME
                              epochs=10,
                              validation_data=(x_test, y_test), 
                              callbacks=[csv_logger])
score = model.evaluate(x_test, y_test, batch_size=32)

print('score[loss, accuracy]:', score)

rec = dict(acc=history.history['acc'], val_acc=history.history['val_acc'],
           loss=history.history['loss'], val_loss=history.history['val_loss'])
np.savez('output.npz', **rec)


import os
print('save the architecture of a model')
json_string = model.to_json()
open(os.path.join('output', 'cnn_model.json'), 'w').write(json_string)
yaml_string = model.to_yaml()
open(os.path.join('output', 'cnn_model.yaml'), 'w').write(yaml_string)
print('save weights')
model.save_weights(os.path.join('output', 'cnn_model_weights.hdf5'))
# model.to