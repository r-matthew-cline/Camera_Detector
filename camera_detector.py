##############################################################################
##
## camera_detector.py
##
## @author: Matthew Cline
## @version: 20180129
##
## Description: A deep conv net to detect which camera was used to capture an
## image. Used for a kaggle competition. 
##
## Data: https://www.kaggle.com/c/sp-society-camera-model-identification/data
##
##############################################################################

import tensorflow as tf
import numpy as np
import pillow
import scipy.misc as misc
import os
import sys
import pickle

######## PLACE TO SAVE THE MODEL AFTER TRAINING ########
modelFn = os.path.normpath('models/tensorflow/camera_detector.ckpt')
if not os.path.exists(os.path.normpath('models/tensorflow')):
    os.makedirs('models/tensorflow')

####### SET UP LOGGING DIRECTORY FOR TENSORBOARD #######
logFn = os.path.normpath('models/tensorflow/logs/camera_detector')
if not os.path.exists(os.path.normpath('models/tensorflow/logs')):
    os.makedirs('models/tensorflow/logs')

######## LOAD IN THE DATA #######
trainImgs = pickle.load(open("trainImages.p", "rb"))
trainLabels = pickle.load(open("trainLabels.p", "rb"))
valImgs = pickle.load(open("valImages.p", "rb"))
valLabels = pickle.load(open("valLabels.p", "rb"))

sess = tf.InteractiveSession()
batchSize = 100
n_iters = 1000
n_classes = 10
learnRate = 0.001

with tf.name_scope('inputs'):
    x = tf.placeholder(tf.float32, [None, 256, 256, 3], name='input_batch')
    y = tf.placeholder(tf.int64, [None, 10], name='input_labels_batch')
    tf.summary.image('input_images', x)
    keep = tf.placeholder(tf.float32)
    tf.summary.scalar('dropout_keep_prob', keep)

def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)

def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)

def variable_summaries(var):
    with tf.name_scope('summaries'):
        mean = tf.reduce_mean(var)
        tf.summary.scalar('mean', mean)
        with tf.name_scope('stddev'):
            stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
        tf.summary.scalar('stddev', stddev)
        tf.summary.scalar('max', tf.reduce_max(var))
        tf.summary.scalar('min', tf.reduce_min(var))
        tf.summary.histogram('histogram', var)


def conv_layer(input_tensor, filter_height, filter_width, input_dim, output_dim, layer_name, activation = tf.nn.leaky_relu):
    with tf.name_scope(layer_name):
        with tf.name_scope('weights'):
            weights = weight_variable([filter_height, filter_width, input_dim, output_dim])
            variable_summaries(weights)
        with tf.name_scope('biases'):
            biases = bias_variable([output_dim])
            variable_summaries(biases)
        with tf.name_scope('convolution'):
            conv = tf.nn.conv2d(input_tensor, weights, strides=[1, 1, 1, 1], padding='SAME') + biases
        with tf.name_scope('activation'):
            activations = activation(conv, name='activation')
            tf.summary.histogram('activations', activations)
    return activations

def fc_layer(input_tensor, input_dim, output_dim, layer_name, activation=tf.nn.leaky_relu):
    with tf.name_scope(layer_name):
        with tf.name_scope('weights'):
            weights = weight_variable([input_dim, output_dim])
            variable_summaries(weights)
        with tf.name_scope('biases'):
            biases = bias_variable([output_dim])
            variable_summaries(biases)
        with tf.name_scope('sumation'):
            preactivate = tf.matmul(input_tensor, weights) + biases
            tf.summary.histogram(preactivate)
        activations = activation(preactivate, name='activation')
        tf.summary.histogram('activations', activations)
        return activations    

######## DEFINE THE MODELS STRUCTURE #######
conv1 = conv_layer(x, 3, 3, 3, 128, 'conv1')
conv1 = tf.nn.dropout(conv1, keep)
conv2 = conv_layer(conv1, 4, 4, 128, 64, 'conv2')
conv2 = tf.nn.dropout(conv2, keep)
conv3 = conv_layer(conv2, 3, 3, 64, 32, 'conv3')
conv3 = tf.nn.dropout(conv3, keep)
conv4 = conv_layer(conv3, 4, 4, 32, 16, 'conv4')
conv4 = tf.nn.dropout(conv4, keep)

with tf.name_scope('reshape'):
    flat = tf.reshape(conv4, [-1, 256 * 256 * 16])

fc1 = fc_layer(flat, 256*256*16, 1024, 'fc1')
fc1 = tf.nn.dropout(fc1, keep)
fc2 = fc_layer(fc1, 1024, 512, 'fc2')
fc2 = tf.nn.dropout(fc2, keep)
fc3 = fc_layer(fc2, 512, 256, 'fc3')
fc3 = tf.nn.dropout(fc3, keep)
fc4 = fc_layer(fc3, 256, 128, 'fc4')
fc4 = tf.nn.dropout(fc4, keep)
fc5 = fc_layer(fc4, 128, 32, 'fc5')
fc5 = tf.nn.dropout(fc5, keep)
out = fc_layer(fc5, 32, n_classes, 'out', activation=tf.identity)

with tf.name_scope('cross_entropy'):
    with tf.name_scope('total'):
        cross_entropy = tf.losses.softmax_cross_entropy(labels=y, logits=out)
tf.summary.scalar('cross_entropy', cross_entropy)

with tf.name_scope('train'):
    train_step = tf.train.AdamOptimizer(learnRate).minimize(cross_entropy)

with tf.name_scope('accuracy'):
    with tf.name_scope('correct_prediction'):
        correct_prediction = tf.equal(tf.argmax(out, 1), tf.argmax(y, 1))
    with tf.name_scope('accuracy'):
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
tf.summary.scalar('accuracy', accuracy)

merged = tf.summary.merge_all()
train_writer = tf.summary.FileWriter(logFn + '/train', sess.graph)
test_writer = tf.summary.FileWriter(logFn + '/test')
tf.global_variables_initializer().run()

if len(sys.argv) < 2:
    print("Please specify a task for the network from the following options:\n     train - start training the model from scratch\n     continue - continue training the model from the last checkpoint\n     validate - test the model's accuracy on the validation set\n\n")
    sys.exit(1)

if sys.argv[1] == 'train':
    for i in range(n_iters):
        for j in range(trainImgs.shape[0] / batchSize + 1):
            batchImgs = []
            for path in trainImgs[j*batchSize:(j+1)*batchSize]:
                tempImg = misc.imread(path)
                batchImgs.append(tempImg)
            batchImgs = np.array(batchImgs)
            batchLabels = trainLabels[j*batchSize:(j+1)*batchSize]
            summary, _ = sess.run([merged, train_step], feed_dict={x: batchImgs, y: batchLabels, keep: 0.7})
            train_writer.add_summary(summary, (i + 1) * (j + 1))
    train_writer.close()
elif sys.argv[1] == 'continue':
    pass
elif sys.argv[1] == 'validate':
    pass
else:
    print("Please specify a task for the network from the following options:\n     train - start training the model from scratch\n     continue - continue training the model from the last checkpoint\n     validate - test the model's accuracy on the validation set\n\n")
    sys.exit(1)