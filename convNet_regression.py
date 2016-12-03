"""
Class for convnets for mnist
"""
import tensorflow as tf
import numpy as np

class ConvNet():
    def __init__(self, learning_rate, batchsz):
        """ 
        Instantiate a ConvNet model.
        :param: learning_rate: learning rate for the SGD optimizer
        """
        self.learning_rate = learning_rate
        self.batchsz = batchsz
        self.x = tf.placeholder(tf.float32, [self.batchsz, 784])
        self.y_ = tf.placeholder(tf.float32, [None, 10])
        self.instantiate_weights()
        self.logits = self.inference()
        #self.softmax, self.loss_val = self.loss()
        self.logits_sum, self.loss_val = self.loss()
        self.train_op = self.train()
        self.accuracy = self.accuracy()

    def instantiate_weights(self):
        """
        Instantiate the network variables
        """
        #Image, labels
        self.x_image = tf.reshape(self.x, [self.batchsz, 28, 28, 1])
        self.y_image = tf.reshape(self.y_, [self.batchsz, 10, 1])
        
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
        logits before final softmax activation).
        """
        #self.x_image = tf.Print(self.x_image, [self.x_image], message="x_image is: ", summarize=100)
        #self.y_image = tf.Print(self.y_image, [self.y_image], message="y_image is: ", summarize=100)
        
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
        #self.h_fc1_drop = tf.Print(self.h_fc1_drop, [self.h_fc1_drop], message="h_fc1_drop is: ", summarize=10)
        
        self.out_logits = tf.matmul(self.h_fc1_drop, self.W_fc2) + self.b_fc2
        #self.out_logits = tf.Print(self.out_logits, [self.out_logits], message="out_logits is: ", summarize=100)
        
        return self.out_logits
    
    def loss(self):
        """
        Build the cross-entropy loss computation graph.
        """
        softmax = tf.nn.softmax(self.logits)
        #softmax = tf.Print(softmax, [softmax], message="softmax is: ", summarize=10)
        
        cross_entropy = tf.reduce_mean(-tf.reduce_sum(self.y_ * tf.log(softmax ), 1), 0)  
        
        #Regression loss
        #softMaxSum = tf.reduce_sum(softmax, 1) 
        #softMaxSum = tf.Print(softMaxSum, [softMaxSum], message="softMaxSum is: ")
               
        logits_sum = tf.reduce_sum(self.logits, 1) 
        logits_sum = tf.Print(logits_sum, [logits_sum], message="logits_sum is: ")
        logits_sum_norm = 10 * (tf.sigmoid(logits_sum))
        logits_sum_norm = tf.Print(logits_sum_norm, [logits_sum_norm], message="logits_sum_norm is: ")

        arr = np.array( [ float(i) for i in range(0,10) ] )
        literal = tf.constant(arr, shape=[10, 1],  dtype=tf.float32)
        labels = tf.matmul(self.y_, literal)

        losses = labels - logits_sum_norm
        loss = tf.abs(tf.reduce_mean(losses))
        #_ = tf.Print(loss, [loss], message="loss is: ")

        #return softmax, cross_entropy
        return logits_sum_norm, loss


    def train(self):
        """
        Build the training operation, using cross-entropy loss and Adam optimizer
        """
        return tf.train.AdamOptimizer(self.learning_rate).minimize(self.loss_val)
    
    def accuracy(self):
        """
        Build the accuaracy metric, using accuracy metric
        """
        #self.correct_prediction = tf.equal(tf.argmax(self.softmax, 1), tf.argmax(self.y_,1))
        self.correct_prediction = tf.equal( tf.cast(tf.round(self.logits_sum), tf.float32), tf.cast(tf.argmax(self.y_, 1), tf.float32))
        return tf.reduce_mean(tf.cast(self.correct_prediction, tf.float32))



    @staticmethod   
    def weight_variable(shape):
        """
        Returns a weight variable of the inputted shape with random initial values.
        :param shape: Size of of the weight variable
        """
        initial = tf.truncated_normal(shape, stddev=0.0001)
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
