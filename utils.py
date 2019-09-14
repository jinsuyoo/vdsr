import tensorflow as tf
import numpy as np
import math

from PIL import Image

from tqdm import tqdm

import os
import h5py


# Read image
def imread(fname):
    return Image.open(fname)


# Save image
def imsave(image, path, fname):
    image = image * 255.
    
    image = Image.fromarray(image.astype('uint8'), mode='YCbCr')
    image = image.convert('RGB')
    
    return image.save(os.path.join(path, fname))


# Save ground truth image, bicubic interpolated image and srcnn image
def save_result(path, gt, bicubic, srcnn, i):
    imsave(gt, path, str(i)+ '_gt.png')
    imsave(bicubic, path, str(i) + '_bicubic.png')
    imsave(srcnn, path, str(i) + '_vdsr.png')


# Return true if the h5 sub-images file is exists
def exist_train_data(datasetname):
    return os.path.exists('{}.h5'.format(datasetname))


# Concatenate Y and CrCb channel
def concat_ycrcb(y, crcb):
    return np.concatenate((y, crcb), axis=2)


def psnr(gt, sr, shave=0, max_val=1.):
    diff = gt[shave:-shave, shave:-shave] - sr[shave:-shave, shave:-shave]
    diff = diff.flatten()
    rmse = math.sqrt(np.mean(diff ** 2))

    return 20 * math.log10(max_val / rmse)


def prepare_data(path, scale, is_valid=False):
    dir_path = os.path.join(os.getcwd(), path)
    path_gt = os.path.join(dir_path, 'gt')
    path_lr = os.path.join(dir_path, 'bicubic_{:d}x'.format(scale))

    # fnames = ['baby_GT.bmp, bird_GT.bmp, ...']
    fnames = os.listdir(path_gt)
    
    inputs = []
    labels = []

    count = 0
    for fname in tqdm(fnames, desc='[*] Generating dataset ... '):
        count += 1
        
        _input = imread(os.path.join(path_lr, fname))
        _label = imread(os.path.join(path_gt, fname))
    
        _input = np.array(_input) / 255.
        _label = np.array(_label) / 255.
        _label = _label[:_label.shape[0] - np.mod(_label.shape[0], scale), :_label.shape[1] - np.mod(_label.shape[1], scale)]
        #_label = _label[:_label.shape[0]//scale, :_label.shape[1]//scale]

        if is_valid:
            h, w, _ = _input.shape

            _input_y = _input[:, :, 0]
            _label_y = _label[:, :, 0]

            _input_y = _input_y.reshape([1, h, w, 1])
            _label_y = _label_y.reshape([1, h, w, 1])

            inputs.append(_input_y)
            labels.append(_label_y)
        
        else:
            inputs.append(_input)
            labels.append(_label)

    if is_valid:
        print('[*] Successfully prepared {:d} valid images !'.format(count))
    else:
        print('[*] Successfully prepared {:d} test images !'.format(count))
        
    return inputs, labels
