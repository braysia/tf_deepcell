from __future__ import division
import os
import numpy as np
from tensorflow.contrib.keras.python.keras.datasets import mnist
from tensorflow.contrib.keras.python.keras.models import Sequential, Model
from tensorflow.contrib.keras.python.keras.layers import Dense, Flatten
from tensorflow.contrib.keras.python.keras.layers import Conv2D, MaxPooling2D
from tensorflow.contrib.keras.python.keras.layers.normalization import BatchNormalization
from tensorflow.contrib.keras import optimizers, callbacks
from tensorflow.contrib.keras import preprocessing
from scipy.ndimage import imread
from patches import extract_patches

x = imread('tests/nuc0/nuclear.png')
y = np.zeros(x.shape)
y0 = imread('tests/nuc0/feature_0.png').astype(bool)
y1 = imread('tests/nuc0/feature_1.png').astype(bool)
y[y0] = 1
y[y1] = 2

ph, pw = 61, 61
assert np.bool(ph & 0x1) and np.bool(pw & 0x1)  # check if odd
nsamples = 50000
frac_test = 0.1
batch_size = 250
nepochs = 10

x_all, y_all = extract_patches(nsamples, x, y, ph, pw)
x_train = x_all[:-int(nsamples * frac_test), :, :]
x_test = x_all[-int(nsamples * frac_test):, :, :]
y_train = y_all[:-int(nsamples * frac_test)]
y_test = y_all[-int(nsamples * frac_test):]
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

opt = optimizers.SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(optimizer=opt, loss='sparse_categorical_crossentropy', metrics=['accuracy'])
csv_logger = callbacks.CSVLogger('training.log')
earlystop = callbacks.EarlyStopping(monitor='val_loss', patience=2)
tensorboard = callbacks.TensorBoard(batch_size=250)
fpath = 'weights.{epoch:02d}-{loss:.2f}-{acc:.2f}-{val_loss:.2f}-{val_acc:.2f}.hdf5'
cp_cb = callbacks.ModelCheckpoint(filepath=fpath, monitor='val_loss', save_best_only=True)


datagen = preprocessing.image.ImageDataGenerator(rotation_range=90, shear_range=0, 
                                                 horizontal_flip=True, vertical_flip=True)
history = model.fit_generator(datagen.flow(x_train, y_train, batch_size=batch_size),
                              steps_per_epoch=len(y_train),  # FIXME
                              epochs=nepochs,
                              validation_data=(x_test, y_test),
                              callbacks=[csv_logger, earlystop, tensorboard, cp_cb])
score = model.evaluate(x_test, y_test, batch_size=32)
print('score[loss, accuracy]:', score)
rec = dict(acc=history.history['acc'], val_acc=history.history['val_acc'],
           loss=history.history['loss'], val_loss=history.history['val_loss'])
np.savez('output.npz', **rec)

output = 'output'
# FIXME: add make dirs
json_string = model.to_json()
open(os.path.join(output, 'cnn_model.json'), 'w').write(json_string)
model.save_weights(os.path.join(output, 'cnn_model_weights.hdf5'))
yaml_string = model.to_yaml()
open(os.path.join(output, 'cnn_model.yaml'), 'w').write(yaml_string)
