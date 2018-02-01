from __future__ import division
import numpy as np


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