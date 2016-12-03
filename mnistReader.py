"""A class for reading MNIST data from files. """

import struct as sys
import numpy as np
import time
import math
import random

np.set_printoptions(edgeitems=56)

class MnistReader:
    """A class for reading MNIST data from files. """
    def __init__(self, path=None):
        self.path = path
        self.X_train = self.readFile('data/train-images-idx3-ubyte', 16, int(6e4), 784, '784B', 'train images')
        self.y_train = self.readFile('data/train-labels-idx1-ubyte', 8, int(6e4), 1, '1B', 'train labels')
        self.X_test = self.readFile('data/t10k-images-idx3-ubyte', 16, int(1e4), 784, '784B', 'test images')
        self.y_test = self.readFile('data/t10k-labels-idx1-ubyte', 8, int(1e4), 1, '1B', 'test labels') 
        
        #Center data mean to zero, set std of each axis to 1
        self.X_train -= np.mean(self.X_train, axis=0)
        self.X_train = np.divide(self.X_train, np.std(self.X_train, axis=0, dtype=np.float64) + 0.000001 )
        self.X_test -= np.mean(self.X_test, axis=0)
        self.X_test /= np.std(self.X_test, axis=0)
       
        #Compute data set properties 
        self.n_train = self.X_train.shape[0]
        self.d_train = self.X_train.shape[1]
        self.bins = 10

    def next_train_batch(self, batchsz):
        """ Returns the next random batch of train images, of batchsize batchsize """
        self.batch_start_index = random.randint(0, self.n_train - batchsz - 1)
        i = self.batch_start_index
        one_hot = self.oneHot(self.y_train[i: i + batchsz], self.bins)
        return (self.X_train[i: i + batchsz], one_hot)
        
    def next_test_batch(self, batchsz):
        """ Returns the next random batch of test images, of batchsize batchsize """
        one_hot = self.oneHot(self.y_test[:batchsz], self.bins)
        return (self.X_test[:batchsz], one_hot)

    def oneHot(self, labels, bins):
        """Returns one-hot encoding of label.
        Assume labels are shape (batch_sz, 1)
        """
        batch_sz = labels.shape[0]
        out = np.zeros([batch_sz, bins])
        #out[label] = 1
        for row in range(0, batch_sz):
            label = int(labels[row])
            out[row][label] = 1
        return out

    def readFile(self, fileName, headerBytes, n, d, unpackParam, label):
        """
        fileName: file name to reader in.
        headerBytes: amount of header bytes in file.
        n: number of items to read in.
        d: length of bytes per item.
        unpackParam: string for sys.unpack to decode bytes.
        label: string to label whether train images, test labels, etc.
        """
        #start = time.time()
        f = open(fileName, 'rb')
        # skip over initial header bytes
        _ = f.read(headerBytes) 
        y_out = np.empty((n,d))
        for i in range(n):
            byts = f.read(d)
            payload = sys.unpack(unpackParam, byts)
            array = np.array(payload, dtype = int)
            y_out[i] = array
        end = time.time()
        #print "The first part of %s are: \n%s." % (label, y_out[:3])
        #print "Elapsed time: %s seconds." % (end - start)
        #print "\n"
        return y_out
