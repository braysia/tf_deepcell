from __future__ import division, print_function
import os
import numpy as np
from tensorflow.contrib.keras import optimizers, callbacks
from tensorflow.contrib.keras.python.keras.preprocessing.image import ImageDataGenerator
from scipy.ndimage import imread
# from tifffile import imread
from patches import extract_patches
from utils import load_model_py, make_outputdir
from os.path import join

FRAC_TEST = 0.1


def prepare_patches(nsamples, frac_test, x, y, ph, pw):
    x_all, y_all = extract_patches(nsamples, x, y, ph, pw)
    x_train = x_all[:-int(nsamples * frac_test), :, :]
    x_test = x_all[-int(nsamples * frac_test):, :, :]
    y_train = y_all[:-int(nsamples * frac_test)]
    y_test = y_all[-int(nsamples * frac_test):]
    x_train = np.expand_dims(x_train, -1)
    x_test = np.expand_dims(x_test, -1)
    return x_train, x_test, y_train, y_test


def define_callbacks(output, batch_size):
    csv_logger = callbacks.CSVLogger(join(output, 'training.log'))
    earlystop = callbacks.EarlyStopping(monitor='val_loss', patience=2)
    tensorboard = callbacks.TensorBoard(batch_size=batch_size)
    fpath = join(output, 'weights.{epoch:02d}-{loss:.2f}-{acc:.2f}-{val_loss:.2f}-{val_acc:.2f}.hdf5')
    cp_cb = callbacks.ModelCheckpoint(filepath=fpath, monitor='val_loss', save_best_only=True)
    return [csv_logger, earlystop, tensorboard, cp_cb]


def train(image_path, labels_path, model_path, output, patchsize=61, nsamples=10000,
          batch_size=32, nepochs=100, frac_test=FRAC_TEST):
    assert np.bool(patchsize & 0x1)  # check if odd
    model = load_model_py(model_path)
    model.summary()

    image, labels = imread(image_path), imread(labels_path).astype(np.uint8)
    x_train, x_test, y_train, y_test = prepare_patches(nsamples, frac_test, image, labels, patchsize, patchsize)

    make_outputdir(output)
    opt = optimizers.SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(optimizer=opt, loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    callbacksets = define_callbacks(output, batch_size)

    datagen = ImageDataGenerator(rotation_range=90, shear_range=0, 
                                 horizontal_flip=True, vertical_flip=True)

    history = model.fit_generator(datagen.flow(x_train, y_train, batch_size=batch_size),
                                  steps_per_epoch=len(y_train),
                                  epochs=nepochs,
                                  validation_data=(x_test, y_test),
                                  callbacks=callbacksets)
    score = model.evaluate(x_test, y_test, batch_size=batch_size)
    print('score[loss, accuracy]:', score)
    rec = dict(acc=history.history['acc'], val_acc=history.history['val_acc'],
               loss=history.history['loss'], val_loss=history.history['val_loss'])
    np.savez(join(output, 'records.npz'), **rec)

    json_string = model.to_json()
    open(join(output, 'cnn_model.json'), 'w').write(json_string)
    model.save_weights(join(output, 'cnn_model_weights.hdf5'))
    yaml_string = model.to_yaml()
    open(join(output, 'cnn_model.yaml'), 'w').write(yaml_string)


def _parse_command_line_args():
    import argparse
    parser = argparse.ArgumentParser(description='predict')
    parser.add_argument('-i', '--image', help='image file path')
    parser.add_argument('-l', '--labels', help='labels file path')
    parser.add_argument('-m', '--model', help='python file path with models')
    parser.add_argument('-o', '--output', default='.', help='output directory')
    parser.add_argument('-n', '--nsamples', type=int, default=100, help='number of samples')
    parser.add_argument('-b', '--batch', type=int, default=32)
    parser.add_argument('-e', '--epoch', type=int, default=50)
    parser.add_argument('-p', '--patch', type=int, default=61,
                        help='pixel size of image patches. make it odd')
    return parser.parse_args()


def _main():
    args = _parse_command_line_args()
    train(args.image, args.labels, args.model, args.output, args.patch,
          args.nsamples, args.batch, args.epoch)

if __name__ == "__main__":
    _main()