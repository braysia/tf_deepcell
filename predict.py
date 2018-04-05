from __future__ import division
import os
import numpy as np
from os.path import join, basename, splitext
from scipy.ndimage import imread
from tensorflow.python.keras import backend
from utils import convert_model_patch2full, load_model_py, make_outputdir
import tifffile as tiff


def predict(img_path, model_path, weight_path):
    x = imread(img_path)
    model = load_model_py(model_path)
    model = convert_model_patch2full(model)
    model.load_weights(weight_path)
    # model.summary()
    # assert False
    evaluate_model = backend.function(
        [model.layers[0].input, backend.learning_phase()],
        [model.layers[-1].output]
        )

    x = np.expand_dims(x, -1)
    x = np.expand_dims(x, 0)
    cc = evaluate_model([x, 0])[0]
    return [cc[0, :, :, i] for i in range(cc.shape[-1])]


def save_output(outputdir, images, pattern):
    make_outputdir(outputdir)
    for num, img in enumerate(images):
        tiff.imsave(join(outputdir, '{0}_l{1}.tif'.format(pattern, num)), img)


def _parse_command_line_args():
    import argparse
    parser = argparse.ArgumentParser(description='predict')
    parser.add_argument('-i', '--image', help='image file path')
    parser.add_argument('-w', '--weight', help='hdf5 file path')
    parser.add_argument('-m', '--model', help='python file path with models')
    parser.add_argument('-o', '--output', default='.', help='output directory')
    return parser.parse_args()


def _main():
    args = _parse_command_line_args()
    images = predict(args.image, args.model, args.weight)
    save_output(args.output, images, splitext(basename(args.image))[0])


if __name__ == "__main__":
    _main()
