#!/usr/bin/env python3
import os.path
import tensorflow as tf
import helper
import warnings
from distutils.version import LooseVersion
import project_tests as tests
import time
import matplotlib.pyplot as plt
import numpy as np
import sys


# Check TensorFlow Version
assert LooseVersion(tf.__version__) >= LooseVersion('1.0'), 'Please use TensorFlow version 1.0 or newer.  You are using {}'.format(tf.__version__)
print('TensorFlow Version: {}'.format(tf.__version__))

# Check for a GPU
if not tf.test.gpu_device_name():
    warnings.warn('No GPU found. Please use a GPU to train your neural network.')
else:
    print('Default GPU Device: {}'.format(tf.test.gpu_device_name()))

# hyperparameters
# some different initialisers
#Kernel_Initializer_stddev = 1e-3
#KERN_INIT = tf.random_normal_initializer(stddev=Kernel_Initializer_stddev)

Kernel_Initializer_gain = 1e-3
KERN_INIT =tf.orthogonal_initializer(gain=Kernel_Initializer_gain)

# regulizer to prevent overfitting
Kernel_L2_regularizer = 1e-4 # must be float - small values of L2 can help prevent overfitting of training data
KERN_REG = tf.contrib.layers.l2_regularizer(Kernel_L2_regularizer)

# parameters to train model
LEARN_RATE = 1e-4
KEEP_PROB = 0.5
EPOCHS = 45
BATCH_SIZE = 15
IMAGE_SHAPE = (160, 576)    # size of image provided to input of net
NUM_CLASSES = 2             # classes for segmenting: ROAD, NOT-ROAD
DATA_DIR = './data'
RUNS_DIR = './runs/'
MODEL_DIR = './model/'
RESULTS_DIR = './results/'

# model name for saving model and data output
MODEL_NAME = 'random_model_EP:'+str(EPOCHS)+'_LR:'+str(LEARN_RATE)+'_KP:'+str(KEEP_PROB)+'_BS:'+str(BATCH_SIZE)+'.ckpt'
RUNS_NAME = RUNS_DIR+'random_model_EP:'+str(EPOCHS)+'_LR:'+str(LEARN_RATE)+'_KP:'+str(KEEP_PROB)+'_BS:'+str(BATCH_SIZE)

# accumulate all losses during training
total_loss = []


def load_vgg(sess, vgg_path):
    """
    Load Pretrained VGG Model into TensorFlow.
    :param sess: TensorFlow Session
    :param vgg_path: Path to vgg folder, containing "variables/" and "saved_model.pb"
    :return: Tuple of Tensors from VGG model (image_input, keep_prob, layer3_out, layer4_out, layer7_out)
    """

    # FCN-8 Encoder
    # load parts of vgg into this new network - weights are frozen but feed more data to force it to learn more
    # pool 3 pool 4 & pool 5

    # variable to hold the name of the net that will be used
    vgg_tag = 'vgg16'

    # create tensor variable names to be used in the FCN
    vgg_input_tensor_name = 'image_input:0' # to feed in image into
    vgg_keep_prob_tensor_name = 'keep_prob:0' # how much data to throw away - and force net to learn
    vgg_layer3_out_tensor_name = 'layer3_out:0' # get out representation of image from vgg
    vgg_layer4_out_tensor_name = 'layer4_out:0' # get out representation of image from vgg
    vgg_layer7_out_tensor_name = 'layer7_out:0' # get out representation of image from vgg

    # load the model and weights from target net and get graph from the files
    tf.saved_model.loader.load(sess, [vgg_tag], vgg_path)

    # get default graph
    graph = tf.get_default_graph()

    # get the layers required for ths FCN
    # input layer tensor
    input_img = graph.get_tensor_by_name(vgg_input_tensor_name)
    # dropout layer tensor
    keep_prob = graph.get_tensor_by_name(vgg_keep_prob_tensor_name)

    # outputs of layers 3,4, and 7
    layer3_out = graph.get_tensor_by_name(vgg_layer3_out_tensor_name)
    layer4_out = graph.get_tensor_by_name(vgg_layer4_out_tensor_name)
    layer7_out = graph.get_tensor_by_name(vgg_layer7_out_tensor_name)

    # tuple of tensors from net model
    return input_img, keep_prob, layer3_out, layer4_out, layer7_out

# run test
tests.test_load_vgg(load_vgg, tf)


def layers(vgg_layer3_out, vgg_layer4_out, vgg_layer7_out, num_classes):

    """
    Create the layers for a fully convolutional network.  Build skip-layers using the vgg layers.
    :param vgg_layer3_out: TF Tensor for VGG Layer 3 output
    :param vgg_layer4_out: TF Tensor for VGG Layer 4 output
    :param vgg_layer7_out: TF Tensor for VGG Layer 7 output
    :param num_classes: Number of classes to classify
    :return: The Tensor for the last layer of output
    """

    # FCN-8 Decoder - upsampling and and adding skip
    # reshape outputs to 1x1 convolutions
    # 1 x 1 convolution layer using layer 7 output
    layer7_conv_1x1 = tf.layers.conv2d(vgg_layer7_out, num_classes, 1, strides=(1,1), padding='same', kernel_regularizer=KERN_REG, kernel_initializer=KERN_INIT, name='layer7_conv_1x1')
    # 1 x 1 convolution layer using layer 4 output
    layer4_conv_1x1 = tf.layers.conv2d(vgg_layer4_out, num_classes, 1, strides=(1,1), padding='same', kernel_regularizer=KERN_REG, kernel_initializer=KERN_INIT, name='layer4_conv_1x1')
    # 1 x 1 convolution layer using layer 3 output
    layer3_conv_1x1 = tf.layers.conv2d(vgg_layer3_out, num_classes, 1, strides=(1,1), padding='same', kernel_regularizer=KERN_REG, kernel_initializer=KERN_INIT, name='layer3_conv_1x1')

    # upsampling is equal to stride of transposed convolution
    # i.e. kernel size = 2 * factor - factor % 2
    # deconvolution/ - upsample layer 7 and add to layer 4 - decoder layer 1
    layer7_upsampling = tf.layers.conv2d_transpose(layer7_conv_1x1, num_classes, kernel_size=4, strides=(2,2), padding='same',kernel_regularizer=KERN_REG,kernel_initializer=KERN_INIT, name='layer7_upsampling')
    # skip layer - add skip to connect from layer 4 of VGG16 to layer 7 - decoder layer 2
    layer4_skip = tf.add(layer7_upsampling, layer4_conv_1x1, name='layer4_skip')

    # deconvolution/ - upsample layer4_skip and add to layer 3 - decoder layer 3
    layer4_upsampling = tf.layers.conv2d_transpose(layer4_skip, num_classes, kernel_size=4, strides=(2,2), padding='same',kernel_regularizer=KERN_REG,kernel_initializer=KERN_INIT, name='layer4_upsampling')
    # skip layer - add skip to connect from layer 3 to layer 4 - decoder layer 4
    layer3_skip = tf.add(layer4_upsampling, layer3_conv_1x1, name='layer3_skip')

    # deconvolution/ - upsample layer3_skip - decoder output layer
    fcn_output = tf.layers.conv2d_transpose(layer3_skip, num_classes, kernel_size=16, strides=(8,8), padding='same',kernel_regularizer=KERN_REG,kernel_initializer=KERN_INIT, name='fcn_output')

    # upsampling skip layers then add 2 > 2 > 8

    return fcn_output

# run test
tests.test_layers(layers)


def optimize(nn_last_layer, correct_label, learning_rate, num_classes):

    # FCN-8 Classifiation & Loss
    # resize 4D to 2D
    # height = no classes, width - no pixels - sets up for softmax

    """
    Build the TensorFLow loss and optimizer operations.
    :param nn_last_layer: TF Tensor of the last layer in the neural network
    :param correct_label: TF Placeholder for the correct label image
    :param learning_rate: TF Placeholder for the learning rate
    :param num_classes: Number of classes to classify
    :return: Tuple of (logits, train_op, cross_entropy_loss)
    """

    # reshape 4D to 2D flattened image
    logits = tf.reshape(nn_last_layer, (-1, num_classes),name='logits')
    labels = tf.reshape(correct_label, (-1, num_classes),name='labels')

    cross_entropy_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=labels))

    # after cross entropy - use atom optimiser
    train_op = tf.train.AdamOptimizer(learning_rate).minimize(cross_entropy_loss)


    return logits, train_op, cross_entropy_loss

# run test
tests.test_optimize(optimize)


def train_nn(sess, epochs, batch_size, get_batches_fn, train_op, cross_entropy_loss, input_image,
             correct_label, keep_prob, learning_rate):
    """
    Train neural network and print out the loss during training.
    :param sess: TF Session
    :param epochs: Number of epochs
    :param batch_size: Batch size
    :param get_batches_fn: Function to get batches of training data.  Call using get_batches_fn(batch_size)
    :param train_op: TF Operation to train the neural network
    :param cross_entropy_loss: TF Tensor for the amount of loss
    :param input_image: TF Placeholder for input images
    :param correct_label: TF Placeholder for label images
    :param keep_prob: TF Placeholder for dropout keep probability
    :param learning_rate: TF Placeholder for learning rate
    """

    # initialise the session
    sess.run(tf.global_variables_initializer())

    for epoch in range(epochs):

        # initialise batch and loss counters
        loss_accum = []
        batch = 1

        # record epoch start time
        epoch_start_time = time.time()
        print ("Epoch {} Start time: {:.3f}".format(epoch, epoch_start_time))


        # get-batches_fn - returns image and label based on batch size
        # loop through images and labels
        for image, label in get_batches_fn(batch_size):

            # run training session
            _, loss = sess.run([train_op, cross_entropy_loss], feed_dict={input_image: image, correct_label: label, learning_rate: LEARN_RATE, keep_prob: KEEP_PROB})

            # add batch loss to running total for this epoch
            loss_accum.append(loss)

            # output batch number and loss
            print('Batch number {} Batch Loss: {:.3f}'.format(batch, loss))
            # increment batch counter
            batch += 1

        epoch_end_time = time.time()
        epoch_loss = sum(loss_accum) / len(loss_accum)
        total_loss.append(epoch_loss)
        print('Total Epoch Time: {:.3f}secs Epoch Loss: {:.3f}'.format((epoch_end_time - epoch_start_time), epoch_loss))

    pass

# run test
tests.test_train_nn(train_nn)


def run():

    tests.test_for_kitti_dataset(DATA_DIR)

    # Download pretrained vgg model
    # helper.maybe_download_pretrained_vgg(DATA_DIR)

    # OPTIONAL: Train and Inference on the cityscapes dataset instead of the Kitti dataset.
    # You'll need a GPU with at least 10 teraFLOPS to train on.
    #  https://www.cityscapes-dataset.com/

    with tf.Session() as sess:
        # Path to vgg model
        vgg_path = os.path.join(DATA_DIR, 'vgg')

        # Create function to get batches
        get_batches_fn = helper.gen_batch_function(os.path.join(DATA_DIR, 'data_road/training'), IMAGE_SHAPE)

        # OPTIONAL: Augment Images for better results
        #  https://datascience.stackexchange.com/questions/5224/how-to-prepare-augment-images-for-neural-network

        # declare placeholders
        correct_label = tf.placeholder(tf.int32, [None, None,None,NUM_CLASSES])
        keep_prob = tf.placeholder(tf.float32)
        learning_rate = tf.placeholder(tf.float32, name='learning_rate')


        # Build NN using load_vgg, layers, and optimize function
        input_image, keep_prob, layer3_out, layer4_out, layer7_out = load_vgg(sess, vgg_path)

        # pass into layers function
        layer_output = layers(layer3_out, layer4_out, layer7_out, NUM_CLASSES)

        # call optimiser to get logits and cross entropy
        logits, train_op, cross_entropy_loss = optimize(layer_output, correct_label, learning_rate, NUM_CLASSES)

        # Train NN using the train_nn function
        train_nn(sess, EPOCHS, BATCH_SIZE, get_batches_fn, train_op, cross_entropy_loss, input_image, correct_label, keep_prob, learning_rate)

        # save the trained model
        saver = tf.train.Saver()

        save_path = saver.save(sess, MODEL_DIR+MODEL_NAME)
        print("Model saved in path: %s" %save_path)

        # Save inference data using helper.save_inference_samples
        helper.save_inference_samples(RUNS_NAME, DATA_DIR, sess, IMAGE_SHAPE, logits, keep_prob, input_image)

        # OPTIONAL: Apply the trained model to a video

        # create inference video like term 1 for advancd lane finding
        # gpu - use 3 classes


if __name__ == '__main__':

    # run training
    run()


    # plot loss graph
    plt.figure(figsize=[8, 6])
    plt.plot(total_loss, 'r', linewidth=1.0)
    plt.legend(['Training Loss'], fontsize=12)
    plt.xlabel('Epochs', fontsize=12)
    plt.ylabel('Loss', fontsize=12)
    plt.ylim(0,1)
    plt.yticks(np.arange(0, 1, step=0.05))
    plt.grid()
    plt.title(MODEL_NAME, fontsize=16)
    plt.savefig(RESULTS_DIR+MODEL_NAME+'_LossCurve.png', transparent=False, bbox_inches='tight')
    plt.show()
