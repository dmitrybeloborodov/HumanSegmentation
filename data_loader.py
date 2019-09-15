import os
import numpy as np
#from torch.utils.data import DataLoader, Dataset
#import random
from PIL import Image
#import matplotlib.pyplot as plt

from utils import resize_and_aug, get_square, normalize, hwc_to_chw

def get_ids(dir):
    '''
    Returns a list of the ids in the directory
    
    Parameters:
    ----------
        dir - path to the directory
    Returns:
    -------
        generator with IDs
    '''
    return (f[:-4] for f in os.listdir(dir))


def split_ids(ids, n=2):
    '''
    Split each id in n, creating n tuples (id, k) for each id
    
    Parameters:
    ----------
        ids - generator of image IDs
        n(int) - number of tuples to produce
    Returns:
    -------
        generator of ID-tuples
    '''
    return ((id, i)  for id in ids for i in range(n))


def to_augged_imgs(ids, dir, suffix, aug, scale):
    '''
    From a list of tuples, returns the correct(top or bottom) augmented img
    
    Parameters:
    ----------
        ids - generator of image IDs
        dir - path to the directory with images
        suffix - file extension
        aug - augmentation to apply (or None to use non-aug images)
    Returns:
    -------
        generator of square images
    '''
    for id, pos in ids:
        im = resize_and_aug(Image.open(dir + id + suffix), aug, scale=scale)
        yield get_square(im, pos)

def get_imgs_and_masks(ids, dir_img, dir_mask, aug, scale):
    '''
    Return all the couples (img, mask)
    
    Parameters:
    ----------
        ids - generator of image IDs
        dir_img - path to the directory of images
        dir_mask - path to the directory of masks
        aug - augmentation to apply (or None to use non-aug images)
        scale - coefficient of scaling
    Returns:
    -------
        couples (image, mask) for all given image IDs
    '''

    imgs = to_augged_imgs(ids, dir_img, '.jpg', aug, scale)
    # need to transform from HWC to CHW
    imgs_switched = map(hwc_to_chw, imgs)
    imgs_normalized = map(normalize, imgs_switched)

    masks = to_augged_imgs(ids, dir_mask, '.png', aug, scale)
    masks_normalized = map(normalize, masks)

    return zip(imgs_normalized, masks_normalized)


def get_full_img_and_mask(id, dir_img, dir_mask):
    '''
    Returns ONE pair of image and mask
    
    Parameters:
    ----------
        id - image(and mask) ID
        dir_img - path to the directory of images
        dir_mask - path to the directory of masks
    Returns:
    -------
        PIL.Image and np.array of image and mask respectively
    '''
    im = Image.open(dir_img + id + '.jpg')
    mask = Image.open(dir_mask + id + '.png')
    return im, np.array(mask)