
import os
import imp
import numpy as np
from scipy.ndimage import imread as imread0
import tifffile as tiff



def conv_labels2dto3d(labels):
    lbnums = np.unique(labels)
    arr = np.zeros((labels.shape[0], labels.shape[1], len(lbnums)), np.uint8)
    for i in lbnums:
        arr[:, :, i] = labels == i
    return arr


def normalize(orig_img):
    percentile = 99.9
    high = np.percentile(orig_img, percentile)
    low = np.percentile(orig_img, 100-percentile)
    img = np.minimum(high, orig_img)
    img = np.maximum(low, img)
    img = (img - low) / (high - low)
    return img


def make_outputdir(output):
    try:
        os.makedirs(output)
    except:
        pass


def imread_check_tiff(path):
    img = imread0(path)
    if img.dtype == 'object' or path.endswith('tif'):
        img = tiff.imread(path)
    return img


def imread(path):
    if isinstance(path, tuple) or isinstance(path, list):
        st = []
        for p in path:
            st.append(imread_check_tiff(p))
        img = np.dstack(st)
        if img.shape[2] == 1:
            np.squeeze(img, axis=2)
        return img
    else:
        return imread_check_tiff(path)


def parse_image_files(inputs):
    if "/" not in inputs:
        return (inputs, )
    store = []
    li = []
    while inputs:
        element = inputs.pop(0)
        if element == "/":
            store.append(li)
            li = []
        else:
            li.append(element)
    store.append(li)
    return zip(*store)