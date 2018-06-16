"""
keras==2.0.0, 2.2 not working
add weight argument to continue training?
"""

from __future__ import division, print_function
import os
import numpy as np

import utils.model_builder
from utils.objectives import weighted_crossentropy
import utils.metrics
import keras.backend
from keras import callbacks
import keras.layers
import keras.models
import keras.optimizers

from tfutils import imread
from patches import extract_patches, pick_coords, pick_coords_list, extract_patch_list, _extract_patches, PatchDataGeneratorList
from tfutils import make_outputdir, normalize, conv_labels2dto3d
from os.path import join
from tfutils import parse_image_files
from on_the_fly import affine_transform, random_flip, random_rotate, random_illum


FRAC_TEST = 0.1
val_batch_size = 10
augment_pipe = [affine_transform(points=20, distort=3), random_flip(),
                random_rotate(), random_illum(perc=0.25)]


def define_callbacks(output):
    csv_logger = callbacks.CSVLogger(join(output, 'training.log'))
    # earlystop = callbacks.EarlyStopping(monitor='loss', patience=2)
    fpath = join(output, 'weights.{epoch:02d}-{loss:.2f}-{categorical_accuracy:.2f}.hdf5')
    cp_cb = callbacks.ModelCheckpoint(filepath=fpath, monitor='loss', save_best_only=True)
    return [csv_logger, cp_cb]


def train(image_list, labels_list, output, patchsize=256, nsamples=6400,
          batch_size=64, nepochs=10, frac_test=FRAC_TEST):

    model = utils.model_builder.get_model_3_class(patchsize, patchsize, activation=None)
    loss = weighted_crossentropy
    metrics = [keras.metrics.categorical_accuracy,
               utils.metrics.channel_recall(channel=0, name="background_recall"),
               utils.metrics.channel_precision(channel=0, name="background_precision"),
               utils.metrics.channel_recall(channel=1, name="interior_recall"),
               utils.metrics.channel_precision(channel=1, name="interior_precision"),
               utils.metrics.channel_recall(channel=2, name="boundary_recall"),
               utils.metrics.channel_precision(channel=2, name="boundary_precision"),
               ]
    optimizer = keras.optimizers.RMSprop(lr=1e-4)
    model.compile(loss=loss, metrics=metrics, optimizer=optimizer)
    model.summary()

    li_image, li_labels = [], []
    for image_path, labels_path in zip(image_list, labels_list):
        image, labels = imread(image_path), imread(labels_path).astype(np.uint8)
        image = normalize(image)
        if image.ndim == 2:
            image = np.expand_dims(image, -1)
        elif image.ndim == 3:
            image = np.moveaxis(image, 0, -1)
        li_image.append(image)
        li_labels.append(labels)

    num_tests = int(nsamples * FRAC_TEST)
    ecoords = pick_coords_list(nsamples, li_labels, patchsize, patchsize)
    li_labels = [conv_labels2dto3d(lb) for lb in li_labels]
    ecoords_tests, ecoords_train = ecoords[:num_tests], ecoords[num_tests:]
    x_tests, y_tests = extract_patch_list(li_image, li_labels, ecoords_tests, patchsize, patchsize)

    make_outputdir(output)
    callbacks = define_callbacks(output)

    datagen = PatchDataGeneratorList(augment_pipe)

    history = model.fit_generator(
        generator=datagen.flow(li_image, li_labels, ecoords_train, patchsize, patchsize, batch_size=batch_size, shuffle=True),
        steps_per_epoch=int(nsamples/batch_size),
        epochs=nepochs,
        validation_data=(x_tests, y_tests),
        # validation_steps=len(ecoords_train)/batch_size,
        validation_steps=len(ecoords_train),
        callbacks=callbacks,
        verbose=1
    )

    # score = model.evaluate(x_tests, y_tests, batch_size=batch_size)
    rec = dict(zip(model.metrics_names, [history.history[i] for i in model.metrics_names]))
    np.savez(join(output, 'records.npz'), **rec)

    json_string = model.to_json()
    open(join(output, 'cnn_model.json'), 'w').write(json_string)
    model.save_weights(join(output, 'cnn_model_weights.hdf5'))
    yaml_string = model.to_yaml()
    open(join(output, 'cnn_model.yaml'), 'w').write(yaml_string)


def _parse_command_line_args():
    """
    image:  Path to a tif or png file (e.g. data/nuc0.png).
            To pass multiple image files (size can be varied), use syntax like
            "-i im0.tif / im1.tif / im2.tif", and pass the same number of labels.
    labels: (e.g. data/labels0.tif)
    model:  path to a python file describing a model (e.g. data/tests_model.py)
            or weights.*.hdf5 produced by ModelCheckPoint.
            To resume training, pass the hdf5 file.
    n:      A number of pixels for training. Use a large number (like 1,000,000)
    batch:  Typically 128-512?
    """

    import argparse
    parser = argparse.ArgumentParser(description='predict')
    parser.add_argument('-i', '--image', help='image file path', nargs="*")
    parser.add_argument('-l', '--labels', help='labels file path', nargs="*")
    parser.add_argument('-o', '--output', default='.', help='output directory')
    parser.add_argument('-n', '--nsamples', type=int, default=100, help='number of samples')
    parser.add_argument('-b', '--batch', type=int, default=64)
    parser.add_argument('-e', '--epoch', type=int, default=50)
    parser.add_argument('-p', '--patch', type=int, default=61,
                        help='pixel size of image patches. make it odd')
    return parser.parse_args()


def _main():
    args = _parse_command_line_args()
    images = parse_image_files(args.image)[0]
    labels = parse_image_files(args.labels)[0]
    train(images, labels, args.output, args.patch,
          args.nsamples, args.batch, args.epoch)


if __name__ == "__main__":
    _main()
