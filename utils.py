import os
import numpy as np
import random

from glob import glob
from PIL import Image
from datetime import datetime

import matplotlib.pyplot as plt

'''---IMAGE PREPROCESSING UTILITIES---'''

def get_square(img, pos):
    '''
    Extract a top or a bottom square from ndarray shape : (H, W, C))
    
    Parameters:
    ----------
        img(ndarray) - image from which to extract a square
        pos(int) - which square(top or bottom) to extract
    Returns:
    -------
        ndarray - a correct square from an image
    '''
    w = img.shape[1]
    if pos == 0:
        return img[:w, :]
    else:
        return img[-w:, :]

def split_img_into_squares(img):
    '''
    Splits an image into two squares
    
    Parameters:
    ----------
        img(ndarray) - image from which to extract squares
    Returns:
    -------
        top and bottom squares from an image
    '''
    return get_square(img, 0), get_square(img, 1)

def hwc_to_chw(img):
    '''
    Changes image shape from (H, W, C) to (C, H, W)
    
    Parameters:
    ----------
        img(ndarray) - input image
    Returns:
    -------
        ndarray image with swapped axes
    '''
    return np.transpose(img, axes=[2, 0, 1])

def resize_and_aug(pilimg, aug, scale=0.5):
    '''
    Rescale and augment an image
    
    Parameters:
    ----------
        pilimg(PIL.Image) - input image
        aug - augmentation to apply(or None)
        scale(float) - scale factor
    Returns:
    -------
        ndarray augmented image
    '''
    w = pilimg.size[0]
    h = pilimg.size[1]
    newW = int(w * scale)
    newH = int(h * scale)

    img = pilimg.resize((newW, newH))
    if aug is not None:
        img = aug(img)
    return np.array(img, dtype=np.float32)

def batch(iterable, batch_size):
    '''
    Yields lists by batch
    
    Parameters:
    ----------
        iterable - objects of iteration (images as ndarray in our case)
        batch_size(int) - size of batch
    Returns:
    -------
        generated batch lists
    '''
    b = []
    for i, t in enumerate(iterable):
        b.append(t)
        if (i + 1) % batch_size == 0:
            yield b
            b = []

    if len(b) > 0:
        yield b

def split_train_val(dataset, val_percent=0.05):
    '''
    Split dataset into 'train' and 'validation' parts
    
    Parameters:
    ----------
        dataset - data to split
        val_percent(float) - fraction of data to use for validation
    Returns:
    -------
        dictionary with 'train' and 'validation' parts
    '''
    dataset = list(dataset)
    length = len(dataset)
    n = int(length * val_percent)
    random.shuffle(dataset)
    return {'train': dataset[:-n], 'val': dataset[-n:]}


def normalize(x):
    '''
    Normalize input
    
    Parameters:
    ----------
        x(int or ndarray) - input to normalize (from images)
    Returns:
    -------
        normalized input
    '''
    return x / 255

def merge_masks(img1, img2, full_h):
    '''
    Merge two square masks vertically into one of given height
    
    Parameters:
    ----------
        img1(ndarray) - image to extract top part from
        img2(ndarray) - image to extract bottom part from
    Returns:
    -------
        ndarray merged image
    '''
    w = img1.shape[1]

    new = np.zeros((full_h, w), np.float32)
    new[:full_h // 2 + 1, :] = img1[:full_h // 2 + 1, :]
    new[full_h // 2 + 1:, :] = img2[-(full_h // 2 - 1):, :]

    return new

'''---IMAGE VISUALISATION FUNCTION---'''

def plot_img_and_mask(img, mask):
    '''
    Plot an image ang its mask
    
    Parameters:
    ----------
        img(PIL.Image or ndarray) - image
        mask(PIL.Image or ndarray) - mask
    '''
    fig = plt.figure()
    a = fig.add_subplot(1, 2, 1)
    a.set_title('Input image')
    plt.imshow(img)
    plt.axis('off')

    b = fig.add_subplot(1, 2, 2)
    b.set_title('Output mask')
    plt.imshow(mask)
    plt.axis('off')
    plt.show()

'''---TASK SUBMISSION UTILITIES---'''

def generate_html(path_to_data):
    """Generates content of html file and saves it.
    Parameters
    ----------
    path_to_data : str
        Path to data with original images, predicted masks, and cropped according masks images.
    Returns
    -------
    str
        Content of html file.
    """
    html = "\n".join(["<!doctype html>", "<html>", "<head>",
                      "<meta http-equiv='Content-Type' content='text/html; charset=utf-8'>",
                      "<title>Визуализация результатов</title>", "</head>", "<body>",
                      "<table cellspacing='0' cellpadding='5'>"]) + "\n"
    paths_to_imgs = sorted(
        ["/".join(path.split("/")[-2:]) for path in glob(f"{path_to_data}/*_img.jpg")])
    paths_to_masks = sorted(
        ["/".join(path.split("/")[-2:]) for path in glob(f"{path_to_data}/*_pred_mask.png")])
    paths_to_crops = sorted(
        ["/".join(path.split("/")[-2:]) for path in glob(f"{path_to_data}/*_crop.png")])
    for ind, (path_to_img, path_to_mask, path_to_crop) in enumerate(zip(paths_to_imgs,
                                                                        paths_to_masks,
                                                                        paths_to_crops)):
        if not ind % 2:
            html += "<tr>\n"
        html += f"<td width='240' valign='top'><img src='{path_to_img}'"
        html += "alt='Something went wrong.'"
        html += f"height='320' title='Original image:\n{path_to_img}'></td>\n"
        html += f"<td width='240' valign='top'><img src='{path_to_mask}'"
        html += "alt='Something went wrong.'"
        html += "height='320' title='Predicted mask'></td>\n"
        html += f"<td width='240' valign='top'><img src='{path_to_crop}'"
        html += "alt='Something went wrong.'"
        html += "height='320' title='Cropped img according\npredicted mask'></td>\n"
        if not ind % 2:
            html += "<td width='100'></td>\n"
        else:
            html += "</tr>\n"
    date = datetime.today().strftime("%Y-%m-%d-%H.%M.%S")
    html += f"</table>\n<i>The page was generated at {date}</i></body>\n</html>"
    filename = os.path.basename(path_to_data) + ".html"
    path_to_save = os.path.dirname(path_to_data)
    with open(f"{path_to_save}/{filename}", "w") as f:
        f.write(html)

    return html


def get_html(paths_to_imgs, pred_masks, path_to_save="results/test"):
    """Generates html file and saves it.
    Parameters
    ----------
    paths_to_imgs : list[str]
        List of paths to original images.
    pred_masks : list[np.ndarray]
        Predicted masks.
    path_to_save : str
        Path to save source images to put them in html file. Html name is the same as name of the
        last folder on `path_to_save` and is saved on upper level.
    Returns
    -------
    str
        Content of html file.
    """
    #paths_to_imgs = np.array(paths_to_imgs)
    #pred_masks = np.array(pred_masks)

    if not os.path.exists(path_to_save):
        os.makedirs(path_to_save)

    #order = np.argsort(paths_to_imgs)
    #paths_to_imgs = paths_to_imgs[order]
    #pred_masks = pred_masks[order]

    for path_to_img, pred_mask in zip(paths_to_imgs, pred_masks):
        img_id = path_to_img.split("/")[-1].split(".")[0]
        img = np.array(Image.open(path_to_img))
        Image.fromarray(img).save(f"{path_to_save}/{img_id}_img.jpg")
        Image.fromarray((pred_mask * 255).astype(np.uint8)).save(f"{path_to_save}/{img_id}_pred_mask.png")
        crop_img = img.copy()
        crop_img[pred_mask == 0] = 0
        Image.fromarray(crop_img).save(f"{path_to_save}/{img_id}_crop.png")

    html = generate_html(path_to_save)

    return html

def encode_rle(mask):
    """Returns encoded mask (run length) as a string.
    Parameters
    ----------
    mask : np.ndarray, 2d
        Mask that consists of 2 unique values: 0 - denotes background, 1 - denotes object.
    Returns
    -------
    str
        Encoded mask.
    Notes
    -----
    Mask should contains only 2 unique values, one of them must be 0, another value, that denotes
    object, could be different from 1 (for example 255).
    """
    pixels = mask.flatten()
    pixels = np.concatenate([[0], pixels, [0]])
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
    runs[1::2] -= runs[::2]

    return ' '.join(str(x) for x in runs)

def decode_rle(rle_mask, shape=(320, 240)):
    """Decodes mask from rle string.
    Parameters
    ----------
    rle_mask : str
        Run length as string formatted.
    shape : tuple of 2 int, optional (default=(320, 240))
        Shape of the decoded image.
    Returns
    -------
    np.ndarray, 2d
        Mask that contains only 2 unique values: 0 - denotes background, 1 - denotes object.
    
    """
    s = rle_mask.split()
    starts, lengths = [np.asarray(x, dtype=int) for x in (s[0:][::2], s[1:][::2])]
    starts -= 1
    ends = starts + lengths
    img = np.zeros(shape[0]*shape[1], dtype=np.uint8)
    for low, high in zip(starts, ends):
        img[low:high] = 1

    return img.reshape(shape)