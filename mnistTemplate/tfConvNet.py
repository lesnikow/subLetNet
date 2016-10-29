"""
tfConvNet.py

Implementation of a convolutional neural network for MNIST in tensorFlow.

Author: Adam Lesnikowski
Verision: Oct 28, 2016
"""
import tensorflow as tf
import numpy as np
import time

class ConvNet():
	def __init__(self, learning_rate):
		"""	
		Instantiate a ConvNet model.
		:param: learning_rate: learning rate for the SGD optimizer
		"""
		self.learning_rate = learning_rate
		
		# Setup placeholders
		self.x = tf.placeholder(tf.float32, [None, 784])
		self.y_ = tf.placeholder(tf.float32, [None, 10])
	
		# Instantiate Net Parameters
		self.instantiate_weights()

		# Build the logits
		self.logits = self.inference()

		# Build the loss computation
		self.softmax, self.loss_val = self.loss()

		# Build the train operation
		self.train_op = self.train()
	
		# Build the accuracy metric for the batch
		self.accuracy = self.accuracy()

	def instantiate_weights(self):
		"""
		Instantiate the network variables
		"""
		#Image
		self.x_image = tf.reshape(self.x, [-1, 28, 28, 1])
	
		#Creates 32 conv filters of size 5x5 
		self.W_conv1 = self.weight_variable( [5, 5, 1, 32] )  
		self.b_conv1 = self.bias_variable( [32] )

		#Creates 64 conv filters of size 5x5, 2 filters for each of the previous 32 conv filters
		self.W_conv2 = self.weight_variable([5, 5, 32, 64]) 
		self.b_conv2 = self.bias_variable( [64] )

		#Images are 64 activations of size 7x7 at this point because of two 2x2 max pool layers
		self.W_fc1 = self.weight_variable([7 * 7 * 64, 1024])
		self.b_fc1 = self.bias_variable([1024])

		self.W_fc2 = self.weight_variable([1024, 10])
		self.b_fc2 = self.bias_variable([10])
		
		
	def inference(self):
		"""
		Build the inference computation graph for the model, going from the input to the output
		logits (before final softmax activation).
		"""
		#First conv and pooling
		self.h_conv1 = tf.nn.relu( self.conv2d(self.x_image, self.W_conv1) + self.b_conv1)
		self.h_pool1 = self.max_pool_2x2(self.h_conv1)
		
		#Second conv and pooling
		self.h_conv2 = tf.nn.relu( self.conv2d(self.h_pool1, self.W_conv2) + self.b_conv2)
		self.h_pool2 = self.max_pool_2x2(self.h_conv2)
		
		#Flatten second hidden pooling layer
		self.h_pool2_flat = tf.reshape(self.h_pool2, [-1, 7*7*64])
		self.h_fc1 = tf.nn.relu(tf.matmul(self.h_pool2_flat, self.W_fc1) + self.b_fc1)
		
		#Dropout after first fully connected layer		
		self.keep_prob = tf.placeholder(tf.float32)
		self.h_fc1_drop = tf.nn.dropout(self.h_fc1, self.keep_prob)

		return  tf.matmul(self.h_fc1_drop, self.W_fc2) + self.b_fc2
 
	
	def loss(self):
		"""
		Build the cross-entropy loss computation graph.
		"""
		softmax = tf.nn.softmax(self.logits)
		cross_entropy = tf.reduce_mean(-tf.reduce_sum(self.y_ * tf.log(softmax ), reduction_indices=[1]))  # self.y_ is the 1-hot enconding of the image label
		return softmax, cross_entropy

	def train(self):
		"""
		Build the training operation, using cross-entropy loss and Adam optimizer
		"""
		return tf.train.AdamOptimizer(self.learning_rate).minimize(self.loss_val)
	
	def accuracy(self):
		"""
		Build the accuaracy metric, using accuracy metric
		"""
		self.correct_prediction = tf.equal(tf.argmax(self.softmax,1), tf.argmax(self.y_,1))
		return tf.reduce_mean(tf.cast(self.correct_prediction, tf.float32))


	@staticmethod	
	def weight_variable(shape):
		"""
		Returns a weight variable of the inputted shape with random initial values.
		:param shape: Size of of the weight variable
		"""
		initial = tf.truncated_normal(shape, stddev=0.1)
		return tf.Variable(initial)
	
	@staticmethod
	def bias_variable(shape):
		"""
		Returns a bias variable of the inputted shape with small constant initial values.
		:param shape: Size of of the weight variable
		"""
		initial = tf.constant(0.1, shape=shape)
		return tf.Variable(initial)

	@staticmethod
	def conv2d(x, W):
		"""
		Returns a conv2d convolution layer 
		:param x: the layer being convoluted on
		:param W: the convolution weights
		"""
		return tf.nn.conv2d(x, W, strides=[1,1,1,1], padding='SAME')
	
	@staticmethod
	def max_pool_2x2(x):
		"""
		Returns a max pool layer.
		:param x: the layer to pool over
		"""
		return tf.nn.max_pool(x, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')


#Main Training Block
if __name__ == "__main__":
	print '\nStarting tfConvNet!\n'
	def helloString():
		""" Prints tf hello string."""
		hw = tf.constant('Loaded tensorflow!')
		sess = tf.Session()
		print(sess.run(hw))

	# Get data
	from tensorflow.examples.tutorials.mnist import input_data
	mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

	#Launch TensorFlow session
	print 'Launching tf session!'
	with tf.Session() as sess:
		# Instantiate Model
		convNet = ConvNet(1e-4)

		# Init net
		sess.run(tf.initialize_all_variables())

		# Train net
		mbatches = input('How many minibatches to train on? ') 
		batchsz = input('How many in each minibatch? ')
		print 'You entered %d minibatches' % mbatches
		print "Training on %d minibatches of size %d..." % ( mbatches, batchsz) 

		time0 = time.time()
		for i in range(mbatches):
			batch_imgs, batch_labs = mnist.train.next_batch(batchsz)
			if i % 50 == 0: 
				train_accuracy = sess.run( convNet.accuracy, feed_dict={convNet.x:batch_imgs, convNet.y_:batch_labs, convNet.keep_prob:1.0})
				print("step %d, training accuracy %g" % (i, train_accuracy) )
			sess.run(convNet.train_op, feed_dict = { convNet.x:batch_imgs, convNet.y_:batch_labs, convNet.keep_prob:0.5 } )
		time1 = time.time()
		print "Training time took %.2f seconds on %d minibatches of size %d" % ( (time1 - time0), mbatches, batchsz) 

		# Measure net accuracy
		print 'Measuring net accuracy...'
		sum_acc = 0
		test_samples = input('How many minibatches to test accuracy on? ')
		for i in range(test_samples):
			batch_imgs, batch_labs = mnist.test.next_batch(batchsz)
			sum_acc += sess.run( convNet.accuracy, feed_dict = { convNet.x:batch_imgs, convNet.y_:batch_labs, convNet.keep_prob:1.0 } )

		print 'Net\'s accuracy on test images is {:.2%}.'.format( sum_acc / test_samples ) 
		print 'Goodbye! \n'
