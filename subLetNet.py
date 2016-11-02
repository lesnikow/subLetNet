from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# Import data
from tensorflow.examples.tutorials.mnist import input_data

import tensorflow as tf

from PIL import Image
import numpy as np
import os, math
import csv

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

# read in training images from given image directory
def readImages(imgDir) :
  row = 171
  col = 256
  images = np.zeros((len(os.listdir(imgDir)), row * col))
  print(images.shape)
  imageIds = []
  index = 0
  for imageName in os.listdir(imgDir) :
      if not imageName.endswith('.jpg') :
          continue
      im = Image.open(imgDir + imageName)
      imgId = imageName.replace('.jpg', '')
      data = np.zeros(row * col)
      arr2d = np.zeros((row, col))
      pixels = im.load()
      for i in range(row):
          for j in range(col):
              r, g, b =  pixels[j, i]
              #print(r, g, b)
              # convert rgb to greyscale
              data[i * col + j] = 0.2989 * r + 0.5870 * g + 0.1140 * b
              #print(i * col + j)
              #print(data[i * col + j])
              arr2d[i, j] = data[i * col + j]
              #print(data)
              #print(arr2d)
      images[index, :] = data[:]
      imageIds.append(imgId)
      index += 1
      print(str(index) + '/' + str(len(os.listdir(imgDir))))
  return imageIds, images

def isNumber(s):
    try:
        float(s)
        return True
    except ValueError:
        return False

# read in all labels from .csv file (including the ones not in training data)
# ans store them in a hashmap, where key is the image id, and value is price bin
def readLabels(labelPath) :
    priceBins = {}
    binSize = 50
    with open(labelPath, 'r') as f:
        reader = csv.reader(f)
        priceList = list(reader)

    for i in range(len(priceList)) :
        if priceList[i][0] == 'id' :
          continue
        id = priceList[i][0]
        price = priceList[i][1]
        price = price.replace('$', '')
        if not isNumber(price) :
           price = 0
        price = float(price)
        priceBins[id] = int(math.floor(price / binSize))
    return priceBins

# read in data
imgDir = 'bos100/'
labelPath = 'labels/bosPrices.csv'

# images
imageIds, images = readImages(imgDir)
# ground truth (labels)
priceBins = readLabels(labelPath)
# find label (prince bin) for 
# each data point in training dataset
labels = []
for id in imageIds :
    labels.append(priceBins[id])

# img attr
row = 171
col = 256
# 1-d size of image
imgLength = row * col
numImages = len(imageIds)
numBins = 10

# ----------------------- START CNN --------------------------------
sess = tf.InteractiveSession()

x = tf.placeholder(tf.float32, [None, imgLength])
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
# 	- for each max_pooling, input size shrinks to half (there are two max poolings)
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

# train and evaluate
cross_entropy = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(y_conv, y_))
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
correct_prediction = tf.equal(tf.argmax(y_conv,1), y_)
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
sess.run(tf.initialize_all_variables())
for i in range(numImages):
	print(str(i) + '/' + str(numImages))
	img = images[i, :]
	label = labels[i]
	labelVector = [0 for element in range(numBins)]
	labelVector[label] = 1
	if i%10 == 0:
	    train_accuracy = accuracy.eval(feed_dict={
	        x:[img], y_: [label], keep_prob: 1.0})
	    print("step %d, training accuracy %g"%(i, train_accuracy))
	print('label: ' + str(label) + ' & prediction: ' + str(tf.argmax(y_conv, 1)))
	train_step.run(feed_dict={x: [img], y_: [label], keep_prob: 0.5})

print("test accuracy %g"%accuracy.eval(feed_dict={
    x: images, y_: labels, keep_prob: 1.0}))

# ----------------------- END --------------------------------
