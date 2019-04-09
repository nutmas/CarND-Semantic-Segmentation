import tensorflow as tf
import os.path
import helper

IMAGE_SHAPE = (160, 576)  # size of image provided to input of net
NUM_CLASSES = 2  # classes for segmenting: ROAD, NOT-ROAD
PREDICT_DIR = './predicts'
MODEL_DIR = './model/random_model_EP:50_LR:0.0001_KP:0.5_BS:15.ckpt'

Kernel_Initializer_gain = 1e-3
KERN_INIT =tf.orthogonal_initializer(gain=Kernel_Initializer_gain)

# regulizer to prevent overfitting
Kernel_L2_regularizer = 1e-4 # must be float - small values of L2 can help prevent overfitting of training data
KERN_REG = tf.contrib.layers.l2_regularizer(Kernel_L2_regularizer)


def load_vgg(sess, vgg_path):

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

def layers(vgg_layer3_out, vgg_layer4_out, vgg_layer7_out, num_classes):

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

    return fcn_output




def predict_freespace(data_path):

    #path to vgg model
    vgg_path = os.path.join('.data', 'vgg')

    with tf.Session() as sess:
        # predict the logits
        # Build NN using load_vgg, layers, and optimize function
        input_image, keep_prob, layer3_out, layer4_out, layer7_out = load_vgg(sess, vgg_path)

        # pass into layers function
        layer_output = layers(layer3_out, layer4_out, layer7_out, NUM_CLASSES)

        # reshape 4D to 2D flattened image
        logits = tf.reshape(layer_output, (-1, NUM_CLASSES), name='logits')

        # restore saved model
        saver=tf.train.Saver()
        saver.restore(sess, MODEL_DIR)

        # Save inference data using helper.save_inference_samples
        helper.save_inference_samples(PREDICT_DIR, data_path, sess, IMAGE_SHAPE, logits, keep_prob, input_image)

if __name__ == '__main__':

    test_data = './data/data_road/testing/image_2'
    predict_freespace(test_data)