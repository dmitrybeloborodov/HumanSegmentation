#import sys
#import os
#from optparse import OptionParser
import numpy as np
import random

import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
from torch import optim
from torchvision import transforms as tf

from eval import eval_net
from models.unet import UNet
from utils import split_train_val, batch
from data_loader import get_ids, split_ids, get_imgs_and_masks


def adjust_learning_rate(optimizer, epoch, lr):
    '''Sets the learning rate to the initial LR decayed by 10 every 30 epochs'''
    lr = lr * (0.1 ** (epoch // 10))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
        
class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def train_net(net,
              epochs=5,
              batch_size=1,
              lr=0.1,
              val_percent=0.05,
              save_cp=True,
              gpu=False,
              img_scale=1.0,
              apply_aug=0,
              colab=True):
    '''
    Trains a given model
    
    Parameters:
    ----------
        net - model to train
        epochs(int) - number of epochs
        batch_size(int) - size of batch
        lr(float) - learning rate
        val_percent(float) - fraction of data to use for validation
        save_cp(bool) - whether to create checkpoints
        gpu(bool) - whether to train on gpu
        img_scale(float) - scale factor
        apply_aug(float) - probability of augmentations application
                           0 to not use augmentations
        colab(bool) - whether the model is trained on Google Colab
    '''

    if colab:
        dir_img = '/content/gdrive/My Drive/MIL Internship/data/train/'
        dir_mask = '/content/gdrive/My Drive/MIL Internship/data/train_mask/'
        dir_checkpoint = '/content/gdrive/My Drive/MIL Internship/checkpoints/'
    else:
        dir_img = 'data/train/'
        dir_mask = 'data/train_mask/'
        dir_checkpoint = 'checkpoints/'

    ids = get_ids(dir_img)
    ids = split_ids(ids)

    iddataset = split_train_val(ids, val_percent)

    print('''
    Starting training:
        Epochs: {}
        Batch size: {}
        Learning rate: {}
        Training size: {}
        Validation size: {}
        Checkpoints: {}
        CUDA: {}
    '''.format(epochs, batch_size, lr, len(iddataset['train']),
               len(iddataset['val']), str(save_cp), str(gpu)))

    N_train = len(iddataset['train'])
    
    random.seed(1337)
    
    #aug_list = [tf.RandomHorizontalFlip(p=1), tf.RandomAffine(degrees=0, translate=(0.3, 0.1))]
    
    optimizer = optim.SGD(net.parameters(),
                          lr=lr,
                          momentum=0.9,
                          weight_decay=0.0005)

    criterion = nn.BCELoss()
    
    max_val_dice = -999

    for epoch in range(epochs):
        print('Starting epoch {}/{}.'.format(epoch + 1, epochs))
        
        adjust_learning_rate(optimizer, epoch, lr)
        losses = AverageMeter()
        net.train()
        
        #if apply_aug:
        #    aug = tf.RandomChoice(aug_list)
        #else:
        #    aug = None

        # reset the generators
        train = get_imgs_and_masks(iddataset['train'], dir_img, dir_mask, None, img_scale)
        val = get_imgs_and_masks(iddataset['val'], dir_img, dir_mask, None, img_scale)

        epoch_loss = 0

        for i, b in enumerate(batch(train, batch_size)):
            
            if random.random() < apply_aug:
                imgs = np.array([np.fliplr(i[0]) for i in b]).astype(np.float32)
                true_masks = np.array([np.fliplr(i[1]) for i in b])
            else:
                imgs = np.array([i[0] for i in b]).astype(np.float32)
                true_masks = np.array([i[1] for i in b])

            imgs = torch.from_numpy(imgs)
            true_masks = torch.from_numpy(true_masks)
            
            if gpu:
                imgs = imgs.cuda()
                true_masks = true_masks.cuda()
            
            masks_pred = net(imgs)
            masks_probs_flat = masks_pred.view(-1)

            true_masks_flat = true_masks.view(-1)

            loss = criterion(masks_probs_flat, true_masks_flat)
            losses.update(loss)
            epoch_loss += loss.item()
            if i % 100 == 0:
                print('{0:.4f} --- loss: {1:.6f}'.format(i * batch_size / N_train, losses.avg))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        print('Epoch finished ! Loss: {}'.format(epoch_loss / i))

        if 1:
            val_dice = eval_net(net, val, gpu)
            print('Validation Dice Coeff: {}'.format(val_dice))

        #if save_cp:
        #    torch.save(net.state_dict(),
        #               dir_checkpoint + 'CP_hflip{}_score{:.3f}.pth'.format(epoch + 1, val_dice))
        #    print('Checkpoint {} saved !'.format(epoch + 1))
            
        if save_cp and val_dice > max_val_dice:
            max_val_dice = val_dice

            torch.save(net.state_dict(),
                       dir_checkpoint + 'best_model_score{:.3f}.pth'.format(val_dice))
            print('Checkpoint {} saved !'.format(epoch + 1))
            '''torch.save({
                'model': net.state_dict(),
                'epoch': epoch,
                'dice_score': val_dice,
                'train_loss': losses.avg
            }, dir_checkpoint + 'UNet_best_model_w/o augs.pth')'''



def get_args():
    parser = OptionParser()
    parser.add_option('-e', '--epochs', dest='epochs', default=5, type='int',
                      help='number of epochs')
    parser.add_option('-b', '--batch-size', dest='batchsize', default=10,
                      type='int', help='batch size')
    parser.add_option('-l', '--learning-rate', dest='lr', default=0.1,
                      type='float', help='learning rate')
    parser.add_option('-g', '--gpu', action='store_true', dest='gpu',
                      default=False, help='use cuda')
    parser.add_option('-c', '--load', dest='load',
                      default=False, help='load file model')
    parser.add_option('-s', '--scale', dest='scale', type='float',
                      default=0.5, help='downscaling factor of the images')

    (options, args) = parser.parse_args()
    return options

if __name__ == '__main__':
    args = get_args()

    net = UNet(n_channels=3, n_classes=1)

    if args.load:
        net.load_state_dict(torch.load(args.load))
        print('Model loaded from {}'.format(args.load))

    if args.gpu:
        net.cuda()
        # cudnn.benchmark = True # faster convolutions, but more memory

    try:
        train_net(net=net,
                  epochs=args.epochs,
                  batch_size=args.batchsize,
                  lr=args.lr,
                  gpu=args.gpu,
                  img_scale=args.scale)
    except KeyboardInterrupt:
        torch.save(net.state_dict(), 'INTERRUPTED.pth')
        print('Saved interrupt')
        try:
            sys.exit(0)
        except SystemExit:
            os._exit(0)