from __future__ import division
import numpy as np
from tensorflow.python.keras.preprocessing.image import ImageDataGenerator, NumpyArrayIterator, Iterator


def _sample_coords_weighted(num, shape, weights):
    flat_idx = np.arange(shape[0] * shape[1])
    chosen_flat = np.random.choice(flat_idx, num, p=weights/weights.sum())
    return np.unravel_index(chosen_flat, shape)


def _calc_equal_weights(features):
    ap = []
    for i in np.unique(features):
        ap.append((features == i).sum())
    frac = np.array([(np.sum(ap) - i)/np.sum(ap) for i in ap])
    prob_2d = np.zeros(features.shape)
    for i in np.unique(features):
        prob_2d[features == i] = frac[i]
    return prob_2d


def pick_coords(num, features, patch_h, patch_w):
    """
    features: img with labels
    """
    prob_2d = _calc_equal_weights(features.astype(np.uint8))
    _ph, _pw = int(np.floor(patch_h/2)), int(np.floor(patch_w/2))
    perim = np.zeros(prob_2d.shape, dtype=np.bool)
    perim[_ph:-_ph, _pw:-_pw] = True
    prob_2d[~perim] = 0
    return _sample_coords_weighted(num, features.shape, prob_2d.flatten())


def _pick_coords(num, features, patch_h, patch_w):
    """
    features: img with labels
    """
    prob_2d = _calc_equal_weights(features.astype(np.uint8))
    _ph, _pw = int(np.floor(patch_h/2)), int(np.floor(patch_w/2))
    perim = np.zeros(prob_2d.shape, dtype=np.bool)
    perim[_ph:-_ph, _pw:-_pw] = True
    prob_2d[~perim] = 0
    return _sample_coords_weighted(num, features.shape, prob_2d.flatten())



def extract_patches(num, x, y, patch_h, patch_w):
    """
    x: input images
    y: feature image with labels

    Sample many windows from a large image. It will correct for labels unbalance.
    (If there are less labels for cell boundaries, it increases the sampling probability)
    """
    coords = pick_coords(num, y, patch_h, patch_w)
    h, w = int(np.floor(patch_h/2)), int(np.floor(patch_w/2))
    xstack = np.zeros((num, patch_h, patch_w), np.float32)
    ystack = np.zeros(num)
    for n, (ch, cw) in enumerate(zip(*coords)):
        xstack[n, :, :] = x[ch-h:ch+h+1, cw-w:cw+w+1]
        ystack[n] = y[ch, cw]
    return xstack, ystack


def _extract_patches(x, y, coords, patch_h, patch_w):
    """
    x: input images
    y: feature image with labels

    Sample many windows from a large image. It will correct for labels unbalance.
    (If there are less labels for cell boundaries, it increases the sampling probability)
    """
    h, w = int(np.floor(patch_h/2)), int(np.floor(patch_w/2))
    xstack = np.zeros((len(coords), patch_h, patch_w), np.float32)
    ystack = np.zeros(len(coords))
    for n, (ch, cw) in enumerate(coords):
        xstack[n, :, :] = x[ch-h:ch+h+1, cw-w:cw+w+1]
        ystack[n] = y[ch, cw]
    return xstack, ystack


class PatchDataGenerator(ImageDataGenerator):
    def flow(self, x, y, coords, patch_h, patch_w, batch_size=32, shuffle=True, seed=None,
             save_to_dir=None, save_prefix='', save_format='png'):
        return CropIterator(
            x, y, self, coords=coords, patch_h=patch_h, patch_w=patch_w,
            batch_size=batch_size, shuffle=shuffle, seed=seed,
            data_format=self.data_format,
            save_to_dir=save_to_dir, save_prefix=save_prefix, save_format=save_format)


class CropIterator(NumpyArrayIterator):
    def __init__(self, *args, **kwards):
        self.coords = kwards.pop('coords')
        self.patch_h = kwards.pop('patch_h')
        self.patch_w = kwards.pop('patch_w')
        self._x, self._y = args[0].copy(), args[1].copy()
        _args = (np.zeros((len(self.coords), self.patch_h, self.patch_w, 1)), np.zeros(len(self.coords)), args[2])  # to get around init error
        # _args = (np.zeros((1, 1, 1, 1)), np.zeros(1), args[2])  # to get around init error
        super(CropIterator, self).__init__(*_args, **kwards)
        self.n = len(self.coords)

    def next(self):
        with self.lock:
            index_array = next(self.index_generator)
        batch_coords = [self.coords[i] for i in index_array]
        x, y = _extract_patches(self._x[0, :, :, 0], self._y, batch_coords, 61, 61)
        self.x = np.expand_dims(x, -1)
        self.y = y
        index_array = np.arange(len(y))
        return self._get_batches_of_transformed_samples(index_array)
