#!/usr/bin/env python3
import os.path
import tensorflow as tf
import helper
import warnings
from distutils.version import LooseVersion
import project_tests as tests


# Check TensorFlow Version
assert LooseVersion(tf.__version__) >= LooseVersion('1.0'), 'Please use TensorFlow version 1.0 or newer.  You are using {}'.format(tf.__version__)
print('TensorFlow Version: {}'.format(tf.__version__))

# Check for a GPU
if not tf.test.gpu_device_name():
    warnings.warn('No GPU found. Please use a GPU to train your neural network.')
else:
    print('Default GPU Device: {}'.format(tf.test.gpu_device_name()))


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
    vgg_keep_prob_tensor_name = 'keep_prob:0'
    vgg_layer3_out_tensor_name = 'layer3_out:0' # get out representation of image from vgg
    vgg_layer4_out_tensor_name = 'layer4_out:0' # get out representation of image from vgg
    vgg_layer7_out_tensor_name = 'layer7_out:0' # get out representation of image from vgg

    # load the model and weights from target net and get graph from the files
    tf.saved_model.loader.load(sess, [vgg_tag], vgg_path)

    # get default graph
    graph = tf.get_default_graph()

    # get the layers required for ths FCN
    # get input tensor
    w1 = graph.get_tensor_by_name(vgg_input_tensor_name)
    # get dropout layer tensor
    keep_prob = graph.get_tensor_by_name(vgg_keep_prob_tensor_name)

    # get outputs of pol layers 3,4, and 7
    pool3 = graph.get_tensor_by_name(vgg_layer3_out_tensor_name)
    pool4 = graph.get_tensor_by_name(vgg_layer4_out_tensor_name)
    pool5 = graph.get_tensor_by_name(vgg_layer7_out_tensor_name)

    # tuple of tensors from net model
    return w1, keep_prob, pool3, pool4, pool5

# run test
tests.test_load_vgg(load_vgg, tf)


def layers(vgg_layer3_out, vgg_layer4_out, vgg_layer7_out, num_classes):

    # architecture of net
    # FCN-8 Decoder
    # skip layer architeure and upsampling

    """
    Create the layers for a fully convolutional network.  Build skip-layers using the vgg layers.
    :param vgg_layer3_out: TF Tensor for VGG Layer 3 output
    :param vgg_layer4_out: TF Tensor for VGG Layer 4 output
    :param vgg_layer7_out: TF Tensor for VGG Layer 7 output
    :param num_classes: Number of classes to classify
    :return: The Tensor for the last layer of output
    """

    # 1 x 1 convolution of layer 7
    layer7_conv_1x1 = tf.layers.conv2d(vgg_layer7_out, num_classes, 1, padding='same', kernel_regularizer=tf.contrib.layers.l2_regularizer(1e-3))
    # 1 x 1 convolution of layer 4
    layer4_conv_1x1 = tf.layers.conv2d(vgg_layer4_out, num_classes, 1, padding='same', kernel_regularizer=tf.contrib.layers.l2_regularizer(1e-3))
    # 1 x 1 convolution of layer 3
    layer3_conv_1x1 = tf.layers.conv2d(vgg_layer3_out, num_classes, 1, padding='same', kernel_regularizer=tf.contrib.layers.l2_regularizer(1e-3))

    # deconvolution/ - upsample layer 7 and add to layer 4 - decoder layer 1
    layer7_upsample = tf.layers.conv2d_transpose(layer7_conv_1x1, num_classes, 4, strides=(2,2), padding='same',kernel_regularizer=tf.contrib.layers.l2_regularizer(1e-3))
    # skip layer - add layer 4 to layer 7 - decoder layer 2
    layer4_skip = tf.add(layer7_upsample, layer4_conv_1x1)

    # deconvolution/ - upsample layer4_skip and add to layer 3 - decoder layer 3
    layer4_upsample = tf.layers.conv2d_transpose(layer4_skip, num_classes, 4, strides=(2,2), padding='same',kernel_regularizer=tf.contrib.layers.l2_regularizer(1e-3))
    # skip layer - add layer 3 to layer 4 - decoder layer 4
    layer3_skip = tf.add(layer4_upsample, layer3_conv_1x1)

    # deconvolution/ - upsample layer3_skip - decoder output layer
    layer3_upsample = tf.layers.conv2d_transpose(layer3_skip, num_classes, 16, strides=(8,8), padding='same',kernel_regularizer=tf.contrib.layers.l2_regularizer(1e-3))

    # upsampling skip layers then add 2 > 2 > 8

    tf.Print(layer3_upsample, [tf.shape(layer3_upsample)[:]])

    return layer3_upsample

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

    # get logits from flattened image
    logits = tf.reshape(nn_last_layer, (-1, num_classes))
    labels = tf.reshape(correct_label, (-1, num_classes))

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

    # get-batches_fn - returns image and label based on batch size
    for epoch in range(epochs):
        print("Epoch: ", epoch)
        for image, label in get_batches_fn(batch_size):
            # now have training info
            _, loss = sess.run([train_op, cross_entropy_loss], feed_dict={input_image: image, correct_label: label, learning_rate: 1e-5, keep_prob: 0.55})

            #print("Training loss: ", loss[1])



# run test
tests.test_train_nn(train_nn)


def run():

    # define number of epochs


    num_classes = 2
    image_shape = (160, 576)  # KITTI dataset uses 160x576 images
    data_dir = './data'
    runs_dir = './runs'
    tests.test_for_kitti_dataset(data_dir)

    # Download pretrained vgg model
    helper.maybe_download_pretrained_vgg(data_dir)

    # OPTIONAL: Train and Inference on the cityscapes dataset instead of the Kitti dataset.
    # You'll need a GPU with at least 10 teraFLOPS to train on.
    #  https://www.cityscapes-dataset.com/

    with tf.Session() as sess:
        # Path to vgg model
        vgg_path = os.path.join(data_dir, 'vgg')

        # Create function to get batches
        get_batches_fn = helper.gen_batch_function(os.path.join(data_dir, 'data_road/training'), image_shape)

        # OPTIONAL: Augment Images for better results
        #  https://datascience.stackexchange.com/questions/5224/how-to-prepare-augment-images-for-neural-network

        # declare placeholders
        correct_label = tf.placeholder(tf.float32)
        keep_prob = tf.placeholder(tf.float32)
        learning_rate = tf.placeholder(tf.float32)


        # TODO: Build NN using load_vgg, layers, and optimize function
        input_image, keep_prob, layer3_out, layer4_out, layer7_out = load_vgg(sess, vgg_path)

        # pass into layers function
        layer_output = layers(layer3_out, layer4_out, layer7_out, num_classes)

        # call optimiser to get logits and cross entropy
        logits, train_op, cross_entropy_loss = optimize(layer_output, correct_label, learning_rate, num_classes)

        # TODO: Train NN using the train_nn function
        epochs = 6
        batch_size = 1

        train_nn(sess, epochs, batch_size, get_batches_fn, train_op, cross_entropy_loss, input_image, correct_label, keep_prob, learning_rate)

        # TODO: Save inference data using helper.save_inference_samples
        helper.save_inference_samples(runs_dir, data_dir, sess, image_shape, logits, keep_prob, input_image)

        # OPTIONAL: Apply the trained model to a video

        # create inference video like term 1 for advancd lane finding
        # gpu - use 3 classes


if __name__ == '__main__':
    run()
