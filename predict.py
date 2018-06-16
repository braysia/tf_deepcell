from __future__ import division
import os
import numpy as np
from os.path import join, basename, splitext
from tfutils import imread
from tfutils import make_outputdir, normalize
import tifffile as tiff
import utils.model_builder


def predict(img_path, weight_path):
    x = imread(img_path)
    x = normalize(x)

    if x.ndim == 2:
        x = np.expand_dims(x, -1)
    elif x.ndim == 3:
        x = np.moveaxis(x, 0, -1)
    x = np.expand_dims(x, 0)

    model = utils.model_builder.get_model_3_class(x.shape[1], x.shape[2])
    model.load_weights(weight_path)
    predictions = model.predict(x, batch_size=1)


    # return [cc[0, :, :, i] for i in range(cc.shape[-1])]


def save_output(outputdir, images, pattern):
    make_outputdir(outputdir)
    for num, img in enumerate(images):
        tiff.imsave(join(outputdir, '{0}_l{1}.tif'.format(pattern, num)), img)


def _parse_command_line_args():
    import argparse
    parser = argparse.ArgumentParser(description='predict')
    parser.add_argument('-i', '--image', help='image file path')
    parser.add_argument('-w', '--weight', help='hdf5 file path')
    parser.add_argument('-o', '--output', default='.', help='output directory')
    return parser.parse_args()


def _main():
    args = _parse_command_line_args()
    images = predict(args.image, args.weight)
    save_output(args.output, images, splitext(basename(args.image))[0])


if __name__ == "__main__":
    _main()
