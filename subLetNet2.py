"""
subLetNet.py

Deep learning on house share images
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf
import numpy as np
import os, math
import csv
import time
import PIL.Image
import DataReader as dr
import Colors
import ConvNet as cn

#To-Dos:
#1. encapsulate conv net details into a ConvNet object
#2. encapsulate other guys

FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_float('learning_rate', 1e-4, 'Learning Rate for Adam Optimizer.')
tf.app.flags.DEFINE_integer('batch_size', 40, 'Batch size for training.')

IMGS_DIR_TEST = 'bos100/train/'
IMGS_DIR_TRAIN = 'bos100/test/'
LABELS_DIR = 'labels/bos/prices.csv'

# Number of classifcation bins
numBins = 4 

dataReader = dr.DataReader()
colors = Colors.Colors()
convNet = cn.ConvNet(1e-4)

# Read in train, test images
imageIdsTrain, imagesTrain = dataReader.readImages(IMGS_DIR_TRAIN)
imageIdsTest, imagesTest = dataReader.readImages(IMGS_DIR_TEST)

# Read in labels
priceBins = dataReader.readLabels(LABELS_DIR)

# Prices
labelsTrain, labelsTest = [], []
for id in imageIdsTrain:
    labelsTrain.append(priceBins[id])
for id in imageIdsTest:
    labelsTest.append(priceBins[id])

print('training labels are %s' % labelsTrain)

# Image attributes
row = 32
col = 32
imgLength = row * col
numImages = len(imageIdsTrain)


# Net
def weight_variable(shape):
  initial = tf.truncated_normal(shape, stddev=0.1)
  return tf.Variable(initial)

def bias_variable(shape):
  initial = tf.constant(0.1, shape=shape)
  return tf.Variable(initial)

def conv2d(x, W):
  return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def max_pool_2x2(x):
  return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                        strides=[1, 2, 2, 1], padding='SAME')
# Setup net
x = tf.placeholder(tf.float32, [None, None])
y_ = tf.placeholder(tf.int64, [None])

# first convolution layer
W_conv1 = weight_variable([5, 5, 1, 32])
b_conv1 = bias_variable([32])

# shape the input to a 4d tensor
x_image = tf.reshape(x, [-1,row,col,1])

# convolve the image with the weight tensor
h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
h_pool1 = max_pool_2x2(h_conv1)

# second convolution layer
W_conv2 = weight_variable([5, 5, 32, 64])
b_conv2 = bias_variable([64])

# convolve
h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
h_pool2 = max_pool_2x2(h_conv2)

# pooling layer
# this is how the input size is changed after two convolution layers
#   - for each max_pooling, input size shrinks to half (there are two max poolings)
W_fc1 = weight_variable([int(math.ceil(row / 4) * math.ceil(col / 4)) * 64, 1024])
b_fc1 = bias_variable([1024])

h_pool2_flat = tf.reshape(h_pool2, [-1, int(math.ceil(row / 4) * math.ceil(col / 4)) * 64])
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

# dropout
keep_prob = tf.placeholder(tf.float32)
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

# readout(softmax)
W_fc2 = weight_variable([1024, numBins])
b_fc2 = bias_variable([numBins])

y_conv = tf.matmul(h_fc1_drop, W_fc2) + b_fc2
prediction = tf.argmax(y_conv,1)

cross_entropy = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(y_conv, y_))
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
correct_prediction = tf.equal(tf.argmax(y_conv,1), y_)
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

train_step = tf.train.trainAdamOptimizer(1e-4).minimize(cross_entropy)


# Train net
mbatches = input('How many minibatches to train on? ')
batchsz = input('How many in each minibatch? ')
print('You entered %d minibatches' % mbatches)
print("Training on %d minibatches of size %d..." % ( mbatches, batchsz) )

timeStart = time.time()
for i in range(mbatches):
    if i % 50 == 0:
       pass


timeEnd = time.time()

