"""
Baseline for machine learning project on road segmentation.
This simple baseline consits of a CNN with two convolutional+pooling layers with a soft-max loss

Credits: Aurelien Lucchi, ETH Zürich

This was last tested with TensorFlow 1.13.2, which is not completely up to date.
To 'downgrade': pip install --upgrade tensorflow==1.13.2
"""

import gzip
import os
import sys
import urllib
import matplotlib.image as mpimg
from PIL import Image

import code

import tensorflow.python.platform

import numpy
import tensorflow as tf

NUM_CHANNELS = 3  # RGB images
IMG_SIZE = 512
PIXEL_DEPTH = 255
NUM_LABELS = 2
TRAINING_SIZE_TOTAL = 400
TRAINING_SIZE = 384
TESTING_SIZE = 50
VALIDATION_SIZE = 16  # Size of the validation set.
SEED = 66478  # Set to None for random seed.
BATCH_SIZE = 16  # 64
NUM_EPOCHS = 60
RESTORE_MODEL = True  # If True, restore existing model instead of training a new one
RECORDING_STEP = 0

# Set image patch size in pixels
# IMG_PATCH_SIZE should be a multiple of 4
# image size should be an integer multiple of this number!
IMG_PATCH_SIZE = 512

tf.app.flags.DEFINE_string('train_dir', 'my_tmp/segment_aerial_images',
                           """Directory where to write event logs """
                           """and checkpoint.""")
FLAGS = tf.app.flags.FLAGS


# Extract patches from a given image
def img_crop(im, w, h):
    list_patches = []
    imgwidth = im.shape[0]
    imgheight = im.shape[1]
    is_2d = len(im.shape) < 3
    for i in range(0, imgheight, h):
        for j in range(0, imgwidth, w):
            if is_2d:
                im_patch = im[j:j+w, i:i+h]
            else:
                im_patch = im[j:j+w, i:i+h, :]
            list_patches.append(im_patch)
    return list_patches


def extract_data(filename, num_images):
    """Extract the images into a 4D tensor [image index, y, x, channels].
    Values are rescaled from [0, 255] down to [-0.5, 0.5].
    """
    imgs = []
    for i in range(1, num_images+1):
        imageid = "satImage_%.3d" % i
        image_filename = filename + imageid + ".png"
        if os.path.isfile(image_filename):
            print('Loading ' + image_filename)
            img = mpimg.imread(image_filename)
            imgs.append(img)
        else:
            print('File ' + image_filename + ' does not exist')

    num_images = len(imgs)
    IMG_WIDTH = imgs[0].shape[0]
    IMG_HEIGHT = imgs[0].shape[1]
    N_PATCHES_PER_IMAGE = (IMG_WIDTH/IMG_PATCH_SIZE)*(IMG_HEIGHT/IMG_PATCH_SIZE)

    img_patches = [img_crop(imgs[i], IMG_PATCH_SIZE, IMG_PATCH_SIZE) for i in range(num_images)]
    data = [img_patches[i][j] for i in range(len(img_patches)) for j in range(len(img_patches[i]))]

    return numpy.asarray(data)

def extract_test_data(filename, num_images):
    """Extract the images into a 4D tensor [image index, y, x, channels].
    Values are rescaled from [0, 255] down to [-0.5, 0.5].
    """
    imgs = []
    for i in range(1, num_images+1):
        imageid = "test_%.0d" % i
        image_filename = filename + imageid + ".png"
        if os.path.isfile(image_filename):
            print('Loading ' + image_filename)
            img = mpimg.imread(image_filename)
            imgs.append(img)
        else:
            print('File ' + image_filename + ' does not exist')

    num_images = len(imgs)
    IMG_WIDTH = imgs[0].shape[0]
    IMG_HEIGHT = imgs[0].shape[1]
    N_PATCHES_PER_IMAGE = (IMG_WIDTH/IMG_PATCH_SIZE)*(IMG_HEIGHT/IMG_PATCH_SIZE)

    img_patches = [img_crop(imgs[i], IMG_PATCH_SIZE, IMG_PATCH_SIZE) for i in range(num_images)]
    data = [img_patches[i][j] for i in range(len(img_patches)) for j in range(len(img_patches[i]))]

    return numpy.asarray(data)


# Assign a label to a patch v
def value_to_class(v):
    #foreground_threshold = 0.25  # percentage of pixels > 1 required to assign a foreground label to a patch
    #df = numpy.sum(v)
    for i in range(1,IMG_PATCH_SIZE+1):
        for j in range(1, IMG_PATCH_SIZE+1):
            if v[i, j] > 0:  # road
                return 1
            else:  # bgrd
                return 0


# Extract label images
def extract_labels(filename, num_images):
    """Extract the labels into a 1-hot matrix [image index, label index]."""
    gt_imgs = []
    for i in range(1, num_images + 1):
        imageid = "satImage_%.3d" % i
        image_filename = filename + imageid + ".png"
        if os.path.isfile(image_filename):
            print('Loading ' + image_filename)
            img = mpimg.imread(image_filename)
            gt_imgs.append(img)
        else:
            print('File ' + image_filename + ' does not exist')

    num_images = len(gt_imgs)
    gt_patches = [img_crop(gt_imgs[i], IMG_PATCH_SIZE, IMG_PATCH_SIZE) for i in range(num_images)]
    data = numpy.asarray([gt_patches[i][j] for i in range(len(gt_patches)) for j in range(len(gt_patches[i]))])
    #labels = numpy.asarray([value_to_class(data[i]) for i in range(len(data))])
    labels = numpy.asarray([data[i]>0.5 for i in range(len(data))])
    # Convert to dense 1-hot representation.
    return labels.astype(numpy.float32)


def error_rate(predictions, labels):
    """Return the error rate based on dense predictions and 1-hot labels."""
    return 100.0 - (
        100.0 *
        numpy.sum(predictions == labels) /
        predictions.size)


# Write predictions from neural network to a file
def write_predictions_to_file(predictions, labels, filename):
    max_labels = numpy.argmax(labels, 1)
    max_predictions = numpy.argmax(predictions, 1)
    file = open(filename, "w")
    n = predictions.shape[0]
    for i in range(0, n):
        file.write(max_labels(i) + ' ' + max_predictions(i))
    file.close()


# Print predictions from neural network
def print_predictions(predictions, labels):
    max_labels = numpy.argmax(labels, 1)
    max_predictions = numpy.argmax(predictions, 1)
    print(str(max_labels) + ' ' + str(max_predictions))


# Convert array of labels to an image
def label_to_img(imgwidth, imgheight, w, h, labels):
    array_labels = numpy.zeros([imgwidth, imgheight])
    idx = 0
    labels = numpy.reshape(labels, [w, h])
    array_labels[labels>0.5] = 1
    array_labels[labels<=0.5] = 0

    return array_labels


def img_float_to_uint8(img):
    rimg = img - numpy.min(img)
    rimg = (rimg / numpy.max(rimg) * PIXEL_DEPTH).round().astype(numpy.uint8)
    return rimg


def concatenate_images(img, gt_img):
    n_channels = len(gt_img.shape)
    w = gt_img.shape[0]
    h = gt_img.shape[1]
    if n_channels == 3:
        cimg = numpy.concatenate((img, gt_img), axis=1)
    else:
        gt_img_3c = numpy.zeros((w, h, 3), dtype=numpy.uint8)
        gt_img8 = img_float_to_uint8(gt_img)          
        gt_img_3c[:, :, 0] = gt_img8
        gt_img_3c[:, :, 1] = gt_img8
        gt_img_3c[:, :, 2] = gt_img8
        img8 = img_float_to_uint8(img)
        cimg = numpy.concatenate((img8, gt_img_3c), axis=1)
    return cimg

def get_mask_image(img):
    w = img.shape[0]
    h = img.shape[1]
    
    gt_img_3c = numpy.zeros((w, h, 3), dtype=numpy.uint8)
    gt_img8 = img_float_to_uint8(img)          
    gt_img_3c[:, :, 0] = gt_img8
    gt_img_3c[:, :, 1] = gt_img8
    gt_img_3c[:, :, 2] = gt_img8

    return Image.fromarray(gt_img_3c, 'RGB').convert("RGBA")

def make_img_overlay(img, predicted_img):
    w = img.shape[0]
    h = img.shape[1]
    color_mask = numpy.zeros((w, h, 3), dtype=numpy.uint8)
    color_mask[:, :, 0] = predicted_img*PIXEL_DEPTH

    img8 = img_float_to_uint8(img)
    background = Image.fromarray(img8, 'RGB').convert("RGBA")
    overlay = Image.fromarray(color_mask, 'RGB').convert("RGBA")
    new_img = Image.blend(background, overlay, 0.2)
    return new_img


def main(argv=None):  # pylint: disable=unused-argument

    data_dir = 'training/'
    train_data_filename = data_dir + 'images/'
    test_data_filename = 'testing/'
    train_labels_filename = data_dir + 'groundtruth/' 

    # Extract it into numpy arrays.
    train_data = extract_data(train_data_filename, TRAINING_SIZE)
    #test_data = extract_test_data(test_data_filename, TESTING_SIZE)
    train_labels = extract_labels(train_labels_filename, TRAINING_SIZE)
    train_labels_all = extract_labels(train_labels_filename, TRAINING_SIZE_TOTAL)

    num_epochs = NUM_EPOCHS

    # c0 = 0  # bgrd
    # c1 = 0  # road
    # for i in range(len(train_labels)):
    #     if train_labels[i][0] == 1:
    #         c0 = c0 + 1
    #     else:
    #         c1 = c1 + 1
    # print('Number of data points per class: c0 = ' + str(c0) + ' c1 = ' + str(c1))

    # print('Balancing training data...')
    # min_c = min(c0, c1)
    # idx0 = [i for i, j in enumerate(train_labels) if j[0] == 1]
    # idx1 = [i for i, j in enumerate(train_labels) if j[1] == 1]
    # new_indices = idx0[0:min_c] + idx1[0:min_c]
    # print(len(new_indices))
    # print(train_data.shape)
    # train_data = train_data[new_indices, :, :, :]
    # train_labels = train_labels[new_indices]

    train_size = train_labels.shape[0]

    # c0 = 0
    # c1 = 0
    # for i in range(len(train_labels)):
    #     if train_labels[i][0] == 1:
    #         c0 = c0 + 1
    #     else:
    #         c1 = c1 + 1
    # print('Number of data points per class: c0 = ' + str(c0) + ' c1 = ' + str(c1))

    # This is where training samples and labels are fed to the graph.
    # These placeholder nodes will be fed a batch of training data at each
    # training step using the {feed_dict} argument to the Run() call below.
    train_data_node = tf.placeholder(
        tf.float32,
        shape=(BATCH_SIZE, IMG_PATCH_SIZE, IMG_PATCH_SIZE, NUM_CHANNELS))
    train_labels_node = tf.placeholder(tf.float32,
                                       shape=(BATCH_SIZE, IMG_SIZE, IMG_SIZE))
    train_all_data_node = tf.constant(train_data)

    # The variables below hold all the trainable weights. They are passed an
    # initial value which will be assigned when when we call:
    # {tf.initialize_all_variables().run()}
    conv1_weights = tf.Variable(
        tf.truncated_normal([5, 5, NUM_CHANNELS, 32],  # 5x5 filter, depth 32.
                            stddev=0.1,
                            seed=SEED))
    conv1_biases = tf.Variable(tf.zeros([32]))

    conv2_weights = tf.Variable(
        tf.truncated_normal([5, 5, 32, 32],
                            stddev=0.1,
                            seed=SEED))
    conv2_biases = tf.Variable(tf.constant(0.1, shape=[32]))


    conv3_weights = tf.Variable(
        tf.truncated_normal([5, 5, 32, 64],
                            stddev=0.1,
                            seed=SEED))
    conv3_biases = tf.Variable(tf.constant(0.1, shape=[64]))

    conv4_weights = tf.Variable(
        tf.truncated_normal([5, 5, 64, 64],
                            stddev=0.1,
                            seed=SEED))
    conv4_biases = tf.Variable(tf.constant(0.1, shape=[64]))


    conv5_weights = tf.Variable(
        tf.truncated_normal([5, 5, 64, 128],
                            stddev=0.1,
                            seed=SEED))
    conv5_biases = tf.Variable(tf.constant(0.1, shape=[128]))

    conv6_weights = tf.Variable(
        tf.truncated_normal([5, 5, 128, 128],
                            stddev=0.1,
                            seed=SEED))
    conv6_biases = tf.Variable(tf.constant(0.1, shape=[128]))
    

    conv7_weights = tf.Variable(
        tf.truncated_normal([5, 5, 128, 128],
                            stddev=0.1,
                            seed=SEED))
    conv7_biases = tf.Variable(tf.constant(0.1, shape=[128]))

    conv8_weights = tf.Variable(
        tf.truncated_normal([5, 5, 128, 128],
                            stddev=0.1,
                            seed=SEED))
    conv8_biases = tf.Variable(tf.constant(0.1, shape=[128]))


    conv9_weights = tf.Variable(
        tf.truncated_normal([5, 5, 128, 64],
                            stddev=0.1,
                            seed=SEED))
    conv9_biases = tf.Variable(tf.constant(0.1, shape=[64]))

    conv10_weights = tf.Variable(
        tf.truncated_normal([5, 5, 64, 64],
                            stddev=0.1,
                            seed=SEED))
    conv10_biases = tf.Variable(tf.constant(0.1, shape=[64]))


    conv11_weights = tf.Variable(
        tf.truncated_normal([5, 5, 64, 32],
                            stddev=0.1,
                            seed=SEED))
    conv11_biases = tf.Variable(tf.constant(0.1, shape=[32]))

    conv12_weights = tf.Variable(
        tf.truncated_normal([5, 5, 32, 32],
                            stddev=0.1,
                            seed=SEED))
    conv12_biases = tf.Variable(tf.constant(0.1, shape=[32]))


    conv_out_weights = tf.Variable(
        tf.truncated_normal([5, 5, 32, 1],
                            stddev=0.1,
                            seed=SEED))
    conv_out_biases = tf.Variable(tf.constant(0.1, shape=[1]))




    # fc1_weights = tf.Variable(  # fully connected, 512.
    #     tf.truncated_normal([int(IMG_PATCH_SIZE / 4 * IMG_PATCH_SIZE / 4 * 64), 512],
    #                         stddev=0.1,
    #                         seed=SEED))
    # fc1_biases = tf.Variable(tf.constant(0.1, shape=[512]))

    # fc2_weights = tf.Variable(  # fully connected, 512.
    #     tf.truncated_normal([512, 256],
    #                         stddev=0.1,
    #                         seed=SEED))
    # fc2_biases = tf.Variable(tf.constant(0.1, shape=[256]))


    # fc3_weights = tf.Variable(
    #     tf.truncated_normal([256, NUM_LABELS],
    #                         stddev=0.1,
    #                         seed=SEED))
    # fc3_biases = tf.Variable(tf.constant(0.1, shape=[NUM_LABELS]))

    # Make an image summary for 4d tensor image with index idx
    def get_image_summary(img, idx=0):
        V = tf.slice(img, (0, 0, 0, idx), (1, -1, -1, 1))
        img_w = img.get_shape().as_list()[1]
        img_h = img.get_shape().as_list()[2]
        min_value = tf.reduce_min(V)
        V = V - min_value
        max_value = tf.reduce_max(V)
        V = V / (max_value*PIXEL_DEPTH)
        V = tf.reshape(V, (img_w, img_h, 1))
        V = tf.transpose(V, (2, 0, 1))
        V = tf.reshape(V, (-1, img_w, img_h, 1))
        return V
    
    # Make an image summary for 3d tensor image with index idx
    def get_image_summary_3d(img):
        V = tf.slice(img, (0, 0, 0), (1, -1, -1))
        img_w = img.get_shape().as_list()[1]
        img_h = img.get_shape().as_list()[2]
        V = tf.reshape(V, (img_w, img_h, 1))
        V = tf.transpose(V, (2, 0, 1))
        V = tf.reshape(V, (-1, img_w, img_h, 1))
        return V

    # Get prediction for given input image 
    def get_prediction(img):
        data = numpy.asarray(img_crop(img, IMG_PATCH_SIZE, IMG_PATCH_SIZE))
        data_node = tf.constant(data)
        output = model(data_node)
        output_prediction = s.run(output)
        img_prediction = label_to_img(img.shape[0], img.shape[1], IMG_PATCH_SIZE, IMG_PATCH_SIZE, output_prediction)
        return img_prediction, output_prediction

    # Get a concatenation of the prediction and groundtruth for given input file
    def get_prediction_with_groundtruth(filename, image_idx):

        imageid = "satImage_%.3d" % image_idx
        image_filename = filename + imageid + ".png"
        img = mpimg.imread(image_filename)
        img_prediction, preds = get_prediction(img)
        labels = train_labels_all[image_idx-1]
        err = error_rate(img_prediction, labels)
        tp = numpy.sum((img_prediction==1) & (labels==1))
        fp = numpy.sum(img_prediction==1) - tp
        fn = numpy.sum((img_prediction==0) & (labels==1))
        cimg = concatenate_images(img, img_prediction)

        return cimg, err, img_prediction, tp, fp, fn

    # Get prediction overlaid on the original image for given input file
    def get_prediction_with_overlay(filename, image_idx, img_prediction):

        imageid = "satImage_%.3d" % image_idx
        image_filename = filename + imageid + ".png"
        img = mpimg.imread(image_filename)

        #img_prediction, _ = get_prediction(img)
        oimg = make_img_overlay(img, img_prediction)

        return oimg

    # Get prediction overlaid on the original image for given input file
    def get_prediction_with_overlay_testing(filename, image_idx):

        imageid = "test_%.0d" % image_idx
        image_filename = filename + imageid + ".png"
        img = mpimg.imread(image_filename)

        img_prediction, _ = get_prediction(img)
        oimg = make_img_overlay(img, img_prediction)

        return oimg, img_prediction

    # We will replicate the model structure for the training subgraph, as well
    # as the evaluation subgraphs, while sharing the trainable parameters.
    def model(data, train=False):
        """The Model definition."""
        # 2D convolution, with 'SAME' padding (i.e. the output feature map has
        # the same size as the input). Note that {strides} is a 4D array whose
        # shape matches the data layout: [image index, y, x, depth].
        conv = tf.nn.conv2d(data,
                            conv1_weights,
                            strides=[1, 1, 1, 1],
                            padding='SAME')
        # Bias and rectified linear non-linearity.
        batch_mean, batch_var = tf.nn.moments(conv, [1,2])
        conv = tf.nn.batch_normalization(conv, tf.reshape(batch_mean, [data.shape[0],1,1,-1]), tf.reshape(batch_var, [data.shape[0],1,1,-1]), scale=1, offset=0, variance_epsilon=0.0001)
        relu = tf.nn.leaky_relu(tf.nn.bias_add(conv, conv1_biases))

        conv2 = tf.nn.conv2d(relu,
                            conv2_weights,
                            strides=[1, 1, 1, 1],
                            padding='SAME')
        # Bias and rectified linear non-linearity.
        batch_mean, batch_var = tf.nn.moments(conv2, [1,2])
        conv2 = tf.nn.batch_normalization(conv2, tf.reshape(batch_mean, [data.shape[0],1,1,-1]), tf.reshape(batch_var, [data.shape[0],1,1,-1]), scale=1, offset=0, variance_epsilon=0.0001)
        relu2 = tf.nn.leaky_relu(tf.nn.bias_add(conv2, conv2_biases))


        # Max pooling. The kernel size spec {ksize} also follows the layout of
        # the data. Here we have a pooling window of 2, and a stride of 2.
        pool = tf.nn.max_pool(relu2,
                              ksize=[1, 4, 4, 1],
                              strides=[1, 4, 4, 1],
                              padding='SAME')


        conv3 = tf.nn.conv2d(pool,
                             conv3_weights,
                             strides=[1, 1, 1, 1],
                             padding='SAME')
        batch_mean, batch_var = tf.nn.moments(conv3, [1,2])                     
        conv3 = tf.nn.batch_normalization(conv3, tf.reshape(batch_mean, [data.shape[0],1,1,-1]), tf.reshape(batch_var, [data.shape[0],1,1,-1]), scale=1, offset=0, variance_epsilon=0.0001)                     
        relu3 = tf.nn.leaky_relu(tf.nn.bias_add(conv3, conv3_biases))

        conv4 = tf.nn.conv2d(relu3,
                             conv4_weights,
                             strides=[1, 1, 1, 1],
                             padding='SAME')
        batch_mean, batch_var = tf.nn.moments(conv4, [1,2])                     
        conv4 = tf.nn.batch_normalization(conv4, tf.reshape(batch_mean, [data.shape[0],1,1,-1]), tf.reshape(batch_var, [data.shape[0],1,1,-1]), scale=1, offset=0, variance_epsilon=0.0001)                     
        relu4 = tf.nn.leaky_relu(tf.nn.bias_add(conv4, conv4_biases))


        pool2 = tf.nn.max_pool(relu4,
                               ksize=[1, 4, 4, 1],
                               strides=[1, 4, 4, 1],
                               padding='SAME')


        conv5 = tf.nn.conv2d(pool2,
                             conv5_weights,
                             strides=[1, 1, 1, 1],
                             padding='SAME')
        batch_mean, batch_var = tf.nn.moments(conv5, [1,2])                     
        conv5 = tf.nn.batch_normalization(conv5, tf.reshape(batch_mean, [data.shape[0],1,1,-1]), tf.reshape(batch_var, [data.shape[0],1,1,-1]), scale=1, offset=0, variance_epsilon=0.0001)                     
        relu5 = tf.nn.leaky_relu(tf.nn.bias_add(conv5, conv5_biases))

        conv6 = tf.nn.conv2d(relu5,
                             conv6_weights,
                             strides=[1, 1, 1, 1],
                             padding='SAME')
        batch_mean, batch_var = tf.nn.moments(conv6, [1,2])                     
        conv6 = tf.nn.batch_normalization(conv6, tf.reshape(batch_mean, [data.shape[0],1,1,-1]), tf.reshape(batch_var, [data.shape[0],1,1,-1]), scale=1, offset=0, variance_epsilon=0.0001)                     
        relu6 = tf.nn.leaky_relu(tf.nn.bias_add(conv6, conv6_biases))

        pool3 = tf.nn.max_pool(relu6,
                               ksize=[1, 4, 4, 1],
                               strides=[1, 4, 4, 1],
                               padding='SAME')

        up = tf.keras.layers.UpSampling2D(size=(4,4))(pool3)      

        conv7 = tf.nn.conv2d(up,
                             conv7_weights,
                             strides=[1, 1, 1, 1],
                             padding='SAME')
        batch_mean, batch_var = tf.nn.moments(conv7, [1,2])                     
        conv7 = tf.nn.batch_normalization(conv7, tf.reshape(batch_mean, [data.shape[0],1,1,-1]), tf.reshape(batch_var, [data.shape[0],1,1,-1]), scale=1, offset=0, variance_epsilon=0.0001)                     
        relu7 = tf.nn.leaky_relu(tf.nn.bias_add(conv7, conv7_biases))

        conv8 = tf.nn.conv2d(relu7,
                             conv8_weights,
                             strides=[1, 1, 1, 1],
                             padding='SAME')
        batch_mean, batch_var = tf.nn.moments(conv8, [1,2])                     
        conv8 = tf.nn.batch_normalization(conv8, tf.reshape(batch_mean, [data.shape[0],1,1,-1]), tf.reshape(batch_var, [data.shape[0],1,1,-1]), scale=1, offset=0, variance_epsilon=0.0001)                     
        relu8 = tf.nn.leaky_relu(tf.nn.bias_add(conv8, conv8_biases))

        up2 = tf.keras.layers.UpSampling2D(size=(4,4))(relu8)



        conv9 = tf.nn.conv2d(up2,
                             conv9_weights,
                             strides=[1, 1, 1, 1],
                             padding='SAME')
        batch_mean, batch_var = tf.nn.moments(conv9, [1,2])                     
        conv9 = tf.nn.batch_normalization(conv9, tf.reshape(batch_mean, [data.shape[0],1,1,-1]), tf.reshape(batch_var, [data.shape[0],1,1,-1]), scale=1, offset=0, variance_epsilon=0.0001)                     
        relu9 = tf.nn.leaky_relu(tf.nn.bias_add(conv9, conv9_biases))

        conv10 = tf.nn.conv2d(relu9,
                             conv10_weights,
                             strides=[1, 1, 1, 1],
                             padding='SAME')
        batch_mean, batch_var = tf.nn.moments(conv10, [1,2])                     
        conv10 = tf.nn.batch_normalization(conv10, tf.reshape(batch_mean, [data.shape[0],1,1,-1]), tf.reshape(batch_var, [data.shape[0],1,1,-1]), scale=1, offset=0, variance_epsilon=0.0001)                     
        relu10 = tf.nn.leaky_relu(tf.nn.bias_add(conv10, conv10_biases))

        up3 = tf.keras.layers.UpSampling2D(size=(4,4))(relu10)  


        conv11 = tf.nn.conv2d(up3,
                             conv11_weights,
                             strides=[1, 1, 1, 1],
                             padding='SAME')
        batch_mean, batch_var = tf.nn.moments(conv11, [1,2])                     
        conv11 = tf.nn.batch_normalization(conv11, tf.reshape(batch_mean, [data.shape[0],1,1,-1]), tf.reshape(batch_var, [data.shape[0],1,1,-1]), scale=1, offset=0, variance_epsilon=0.0001)                     
        relu11 = tf.nn.leaky_relu(tf.nn.bias_add(conv11, conv11_biases))

        conv12 = tf.nn.conv2d(relu11,
                             conv12_weights,
                             strides=[1, 1, 1, 1],
                             padding='SAME')
        batch_mean, batch_var = tf.nn.moments(conv12, [1,2])                     
        conv12 = tf.nn.batch_normalization(conv12, tf.reshape(batch_mean, [data.shape[0],1,1,-1]), tf.reshape(batch_var, [data.shape[0],1,1,-1]), scale=1, offset=0, variance_epsilon=0.0001)                     
        relu12 = tf.nn.leaky_relu(tf.nn.bias_add(conv12, conv12_biases))


        conv_out = tf.nn.conv2d(relu12,
                             conv_out_weights,
                             strides=[1, 1, 1, 1],
                             padding='SAME')             
        out = tf.nn.leaky_relu(tf.nn.bias_add(conv_out, conv_out_biases))
        out = tf.math.sigmoid(out)
        # Uncomment these lines to check the size of each layer
        # print 'data ' + str(data.get_shape())
        # print 'conv ' + str(conv.get_shape())
        # print 'relu ' + str(relu.get_shape())
        # print 'pool ' + str(pool.get_shape())
        # print 'pool2 ' + str(pool2.get_shape())

        # Reshape the feature map cuboid into a 2D matrix to feed it to the
        # fully connected layers.
      #  pool_shape = pool2.get_shape().as_list()
       # reshape = tf.reshape(
      #      pool2,
       #     [pool_shape[0], pool_shape[1] * pool_shape[2] * pool_shape[3]])
        # Fully connected layer. Note that the '+' operation automatically
        # broadcasts the biases.
       # hidden1 = tf.nn.leaky_relu(tf.matmul(reshape, fc1_weights) + fc1_biases)

       # hidden2 = tf.nn.leaky_relu(tf.matmul(hidden1, fc2_weights) + fc2_biases)
        # Add a 50% dropout during training only. Dropout also scales
        # activations such that no rescaling is needed at evaluation time.
        #if train:
        #    hidden = tf.nn.dropout(hidden, 0.5, seed=SEED)
       # out = tf.matmul(hidden2, fc3_weights) + fc3_biases

        # if train:
        #     summary_id = '_0'
        #     s_data = get_image_summary(data)
        #     tf.summary.image('summary_data' + summary_id, s_data, max_outputs=3)
        #     s_conv = get_image_summary(conv)
        #     tf.summary.image('summary_conv' + summary_id, s_conv, max_outputs=3)
        #     s_pool = get_image_summary(pool)
        #     tf.summary.image('summary_pool' + summary_id, s_pool, max_outputs=3)
        #     s_conv2 = get_image_summary(conv2)
        #     tf.summary.image('summary_conv2' + summary_id, s_conv2, max_outputs=3)
        #     s_pool2 = get_image_summary(pool2)
        #     tf.summary.image('summary_pool2' + summary_id, s_pool2, max_outputs=3)

        return out

    # Training computation: logits + cross-entropy loss.
    logits = model(train_data_node, True)  # BATCH_SIZE*NUM_LABELS
    # print 'logits = ' + str(logits.get_shape()) + ' train_labels_node = ' + str(train_labels_node.get_shape())
    logits = tf.reshape(logits, [BATCH_SIZE, IMG_SIZE, IMG_SIZE])
    #loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels = train_labels_node, logits = logits))
    loss = tf.reduce_mean(tf.nn.l2_loss(train_labels_node - logits))

    tf.summary.scalar('loss', loss)

    all_params_node = [conv1_weights, conv1_biases, conv2_weights, conv2_biases, conv3_weights, conv3_biases, conv4_weights, conv4_biases,
     conv5_weights, conv5_biases, conv6_weights, conv6_biases, conv7_weights, conv7_biases, conv8_weights, conv8_biases,
     conv9_weights, conv9_biases, conv10_weights, conv10_biases, conv11_weights, conv11_biases, conv12_weights, conv12_biases,
     conv_out_weights, conv_out_biases]
    all_params_names = ['conv1_weights', 'conv1_biases', 'conv2_weights', 'conv2_biases', 'conv3_weights', 'conv3_biases', 'conv4_weights', 'conv4_biases',
     'conv5_weights', 'conv5_biases', 'conv6_weights', 'conv6_biases', 'conv7_weights', 'conv7_biases', 'conv8_weights', 'conv8_biases',
     'conv9_weights', 'conv9_biases', 'conv10_weights', 'conv10_biases', 'conv11_weights', 'conv11_biases', 'conv12_weights', 'conv12_biases',
     'conv_out_weights', 'conv_out_biases']
    all_grads_node = tf.gradients(loss, all_params_node)
    all_grad_norms_node = []
    for i in range(0, len(all_grads_node)):
        norm_grad_i = tf.global_norm([all_grads_node[i]])
        all_grad_norms_node.append(norm_grad_i)
        tf.summary.scalar(all_params_names[i], norm_grad_i)
    
    # L2 regularization for the fully connected parameters.
    #regularizers = (tf.nn.l2_loss(fc1_weights) + tf.nn.l2_loss(fc1_biases) +
     #               tf.nn.l2_loss(fc2_weights) + tf.nn.l2_loss(fc2_biases) +
     #               tf.nn.l2_loss(fc3_weights) + tf.nn.l2_loss(fc3_biases))
    # Add the regularization term to the loss.
    #loss += 5e-4 * regularizers

    # Optimizer: set up a variable that's incremented once per batch and
    # controls the learning rate decay.
    batch = tf.Variable(0)
    # Decay once per epoch, using an exponential schedule starting at 0.01.
    learning_rate = tf.train.exponential_decay(
        0.001,                # Base learning rate.
        batch * BATCH_SIZE,  # Current index into the dataset.
        train_size,          # Decay step.
        0.95,                # Decay rate.
        staircase=True)
    # tf.scalar_summary('learning_rate', learning_rate)
    tf.summary.scalar('learning_rate', learning_rate)
    
    # Use simple momentum for the optimization.
    #optimizer = tf.train.MomentumOptimizer(learning_rate, 0.0).minimize(loss, global_step=batch)
    optimizer = tf.train.AdamOptimizer(learning_rate).minimize(loss, global_step = batch)
    # Predictions for the minibatch, validation set and test set.
    train_prediction = logits
    # We'll compute them only once in a while by calling their {eval()} method.
    #train_all_prediction = tf.nn.softmax(model(train_all_data_node))

    # Add ops to save and restore all the variables.
    saver = tf.train.Saver()

    # Create a local session to run this computation.
    with tf.Session() as s:

        if RESTORE_MODEL:
            # Restore variables from disk.
            saver.restore(s, FLAGS.train_dir + "/model.ckpt")
            print("Model restored.")

        else:
            # Run all the initializers to prepare the trainable parameters.
            tf.global_variables_initializer().run()

            # Build the summary operation based on the TF collection of Summaries.
            summary_op = tf.summary.merge_all()
            summary_writer = tf.summary.FileWriter(FLAGS.train_dir,
                                                   graph=s.graph)

            print('Initialized!')
            # Loop through training steps.
            print('Total number of iterations = ' + str(int(num_epochs * train_size / BATCH_SIZE)))

            training_indices = range(train_size)

            for iepoch in range(num_epochs):

                # Permute training indices
                perm_indices = numpy.random.permutation(training_indices)

                steps_per_epoch = int(train_size / BATCH_SIZE)

                for step in range(steps_per_epoch):

                    offset = (step * BATCH_SIZE) % (train_size - BATCH_SIZE)
                    batch_indices = perm_indices[offset:(offset + BATCH_SIZE)]

                    # Compute the offset of the current minibatch in the data.
                    # Note that we could use better randomization across epochs.
                    batch_data = train_data[batch_indices, :, :, :]
                    batch_labels = train_labels[batch_indices]
                    # This dictionary maps the batch data (as a numpy array) to the
                    # node in the graph is should be fed to.
                    feed_dict = {train_data_node: batch_data,
                                 train_labels_node: batch_labels}

                    if step == 0:
                        summary_str, _, l, lr, predictions = s.run(
                            [summary_op, optimizer, loss, learning_rate, train_prediction],
                            feed_dict=feed_dict)
                        summary_writer.add_summary(summary_str, iepoch * steps_per_epoch)
                        summary_writer.flush()

                        print('Epoch %d' % iepoch)
                        print('Minibatch loss: %.3f, learning rate: %.6f' % (l, lr))
                        print('Minibatch error: %.1f%%' % error_rate(predictions,
                                                                     batch_labels))

                        sys.stdout.flush()
                    else:
                        # Run the graph and fetch some of the nodes.
                        _, l, lr, predictions = s.run(
                            [optimizer, loss, learning_rate, train_prediction],
                            feed_dict=feed_dict)

                # Save the variables to disk.
                save_path = saver.save(s, FLAGS.train_dir + "/model.ckpt")
                print("Model saved in file: %s" % save_path)
        
        print("Running prediction on test set")
        prediction_testing_dir = "predictions_testing/"
        testing_masks_dir = "testing_masks/"
        if not os.path.isdir(prediction_testing_dir):
            os.mkdir(prediction_testing_dir)
        if not os.path.isdir(testing_masks_dir):
            os.mkdir(testing_masks_dir)
        for i in range(1, TESTING_SIZE + 1):
           # pimg, err = get_prediction_with_groundtruth(test_data_filename, i)
           # err_total = err_total + err
           # Image.fromarray(pimg).save(prediction_testing_dir + "prediction_" + str(i) + ".png")
            oimg, img_prediction = get_prediction_with_overlay_testing(test_data_filename, i)
            img_prediction = get_mask_image(img_prediction)
            oimg.save(prediction_testing_dir + "overlay_" + str(i) + ".png")  
            img_prediction.save(testing_masks_dir + "gt_test_" + str(i) + ".png")     
          #  print(err_total/TRAINING_SIZE)


if __name__ == '__main__':
    tf.app.run()
