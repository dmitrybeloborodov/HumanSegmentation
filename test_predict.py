import argparse
import os

import numpy as np
import torch
import torch.nn.functional as F

from PIL import Image

from models.unet import UNet
from utils import resize_and_aug, normalize, split_img_into_squares, hwc_to_chw, merge_masks
from utils import plot_img_and_mask

from torchvision import transforms

def predict_img(net,
                full_img,
                scale_factor=1.0,
                out_threshold=0.5,
                use_gpu=False):
    '''
    Predicts the mask of an image
    
    Parameters:
    ----------
        net - model to use for prediction
        full_img(PIL.Image) - image for prediction
        scale_factor(float) - factor for image scaling
        out_threshold(float) - min probability to consider a pixel white
        use_gpu(bool) - whether to use a gpu
    Returns:
    -------
        mask in ndarray type
    '''

    net.eval()
    img_height = full_img.size[1]
    img_width = full_img.size[0]

    img = resize_and_aug(full_img, aug=None, scale=scale_factor)
    img = normalize(img)

    top_square, bottom_square = split_img_into_squares(img)

    top_square = hwc_to_chw(top_square)
    bottom_square = hwc_to_chw(bottom_square)

    X_top = torch.from_numpy(top_square).unsqueeze(0)
    X_bottom = torch.from_numpy(bottom_square).unsqueeze(0)
    
    if use_gpu:
        X_top = X_top.cuda()
        X_bottom = X_bottom.cuda()

    with torch.no_grad():
        output_top = net(X_top)
        output_bottom = net(X_bottom)

        top_probs = output_top.squeeze(0)
        bottom_probs = output_bottom.squeeze(0)

        tf = transforms.Compose(
            [
                transforms.ToPILImage(),
                transforms.Resize(img_width),
                transforms.ToTensor()
            ]
        )
        
        top_probs = tf(top_probs.cpu())
        bottom_probs = tf(bottom_probs.cpu())

        top_mask_np = top_probs.squeeze().cpu().numpy()
        bottom_mask_np = bottom_probs.squeeze().cpu().numpy()

    full_mask = merge_masks(top_mask_np, bottom_mask_np, img_height)

    return full_mask > out_threshold



def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', '-m', default='MODEL.pth',
                        metavar='FILE',
                        help="Specify the file in which is stored the model"
                             " (default : 'MODEL.pth')")
    parser.add_argument('--input', '-i', metavar='INPUT', nargs='+',
                        help='filenames of input images', required=True)

    parser.add_argument('--output', '-o', metavar='INPUT', nargs='+',
                        help='filenames of ouput images')
    parser.add_argument('--cpu', '-c', action='store_true',
                        help="Do not use the cuda version of the net",
                        default=False)
    parser.add_argument('--viz', '-v', action='store_true',
                        help="Visualize the images as they are processed",
                        default=False)
    parser.add_argument('--no-save', '-n', action='store_true',
                        help="Do not save the output masks",
                        default=False)
    parser.add_argument('--mask-threshold', '-t', type=float,
                        help="Minimum probability value to consider a mask pixel white",
                        default=0.5)
    parser.add_argument('--scale', '-s', type=float,
                        help="Scale factor for the input images",
                        default=0.5)

    return parser.parse_args()

def get_output_filenames(args):
    in_files = args.input
    out_files = []

    if not args.output:
        for f in in_files:
            pathsplit = os.path.splitext(f)
            out_files.append("{}_OUT{}".format(pathsplit[0], pathsplit[1]))
    elif len(in_files) != len(args.output):
        print("Error : Input files and output files are not of the same length")
        raise SystemExit()
    else:
        out_files = args.output

    return out_files

def mask_to_image(mask):
    return Image.fromarray((mask * 255).astype(np.uint8))

if __name__ == "__main__":
    args = get_args()
    in_files = args.input
    out_files = get_output_filenames(args)

    net = UNet(n_channels=3, n_classes=1)

    print("Loading model {}".format(args.model))

    if not args.cpu:
        print("Using CUDA version of the net, prepare your GPU !")
        net.cuda()
        net.load_state_dict(torch.load(args.model))
    else:
        net.cpu()
        net.load_state_dict(torch.load(args.model, map_location='cpu'))
        print("Using CPU version of the net, this may be very slow")

    print("Model loaded !")

    for i, fn in enumerate(in_files):
        print("\nPredicting image {} ...".format(fn))

        img = Image.open(fn)
        if img.size[0] < img.size[1]:
            print("Error: image height larger than the width")

        mask = predict_img(net=net,
                           full_img=img,
                           scale_factor=args.scale,
                           out_threshold=args.mask_threshold,
                           use_gpu=not args.cpu)

        if args.viz:
            print("Visualizing results for image {}, close to continue ...".format(fn))
            plot_img_and_mask(img, mask)

        if not args.no_save:
            out_fn = out_files[i]
            result = mask_to_image(mask)
            result.save(out_files[i])

            print("Mask saved to {}".format(out_files[i]))