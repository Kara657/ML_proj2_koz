#!/usr/bin/env python3

import os
import numpy as np
import matplotlib.image as mpimg
import re
import skimage
from skimage.transform import resize
from matplotlib import pyplot as plt
from PIL import Image

foreground_threshold = 0.25 # percentage of pixels > 1 required to assign a foreground label to a patch

# assign a label to a patch
def patch_to_label(patch):
    df = np.mean(patch)
    if df > foreground_threshold:
        return 1
    else:
        return 0

def pixelwise_to_patchwise(im, idx):
    patch_size = 16
    patchwise_img = np.zeros((im.shape[0],im.shape[1]))
    for j in range(0, im.shape[1], patch_size):
        for i in range(0, im.shape[0], patch_size):
            patch = im[i:i + patch_size, j:j + patch_size]
            label = patch_to_label(patch)
            patchwise_img[i:i+patch_size,j:j+patch_size]= label
    patchwise_testing_dir = "patchwise_testing_masks/"
    if not os.path.isdir(patchwise_testing_dir):
        os.mkdir(patchwise_testing_dir)
        
    Image.fromarray(patchwise_img.astype('uint8')*255).save(patchwise_testing_dir + "patchwise_gt_test_" + str(idx) + ".png")
    return patchwise_img






def mask_to_submission_strings(image_filename, idx):
    """Reads a single image and outputs the strings that should go into the submission file"""
    img_number = int(re.search(r"\d+", image_filename).group(0))
    im = mpimg.imread(image_filename)
    im = resize(im, (608,608))#Resize to original test image size
    patch_size = 16
    patchwise_im = pixelwise_to_patchwise(im, idx)
    for j in range(0, patchwise_im.shape[1], patch_size):
        for i in range(0, patchwise_im.shape[0], patch_size):
            patch = patchwise_im[i:i + patch_size, j:j + patch_size]
            label = patch_to_label(patch)
            yield("{:03d}_{}_{},{}".format(img_number, j, i, label))


def masks_to_submission(submission_filename, *image_filenames):
    """Converts images into a submission file"""
    with open(submission_filename, 'w') as f:
        f.write('id,prediction\n')
        idx=0
        for fn in image_filenames[0:]:
            idx = idx+1
            f.writelines('{}\n'.format(s) for s in mask_to_submission_strings(fn, idx))


if __name__ == '__main__':
    submission_filename = 'submission_final.csv'
    image_filenames = []
    for i in range(1, 51):
        image_filename = 'testing_masks/gt_test_' + '%.0d' % i + '.png'
        print (image_filename)
        image_filenames.append(image_filename)
    masks_to_submission(submission_filename, *image_filenames)
