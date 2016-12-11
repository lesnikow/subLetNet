"""
main.py
Implementation of a convolutional neural network for MNIST in tensorFlow.
Author: Adam Lesnikowski
"""
import sys
sys.path.insert(0, 'modules')
import tensorflow as tf
import numpy as np
import time as ti
import datetime as dt
from colors import Colors
from reader import Reader
from plotter import Plotter
from convNet import ConvNet
from mnistReader import MnistReader
from imageIterator import ImageIterator
from logger import Logger
from tensorflow.examples.tutorials.mnist import input_data

mbatches = int(8.0e4)
batchsz = 128
l_rate = 0.0625e-5
interval = 20

eval_train_accuracy_every = interval
plot_train_accuracy_every = interval
eval_test_accuracy_every = interval
plot_test_accuracy_every = interval

#Main Training Block
if __name__ == "__main__":
    plotter = Plotter()
    print("Reading in data...")
    mnistReader = MnistReader("MNIST_data")
    mnistReaderOrg = input_data.read_data_sets("MNIST_data/", one_hot=True)
    #imageIter = ImageIterator("lon_small/square_28/")
    
    with tf.Session() as sess:
        convNet = ConvNet(l_rate, batchsz)
        train_logger = Logger("Train set")
        test_logger = Logger("Test set")
        sess.run(tf.initialize_all_variables())
        # Train net
        print("Training on %d minibatches of size %d..." % ( mbatches, batchsz) )
        time0 = ti.time()
        for i in range(mbatches):
            batch_imgs, batch_labs = mnistReader.next_train_batch(batchsz)

            loss, _ = sess.run([convNet.loss_val, convNet.train_op],
                                feed_dict = {convNet.x:batch_imgs, convNet.y_:batch_labs, convNet.keep_prob:0.5})

            if i % eval_train_accuracy_every == 0:
                batch_imgs_train, batch_labs_train = mnistReader.next_train_batch(batchsz)
                train_loss, train_accuracy = sess.run( [convNet.loss_val, convNet.accuracy], feed_dict=
                                           {convNet.x:batch_imgs_train, convNet.y_:batch_labs_train, convNet.keep_prob:1.0})
                plotter.x_axis_losses_train.append(i)
                plotter.x_axis_acc_train.append(i)
                plotter.losses_train.append(train_loss)
                plotter.acc_train.append(train_accuracy)
                train_logger.log(dt.datetime.now(), train_loss, train_accuracy, i)

                
                
            if i % eval_test_accuracy_every == 0:
                batch_imgs_test, batch_labs_test = mnistReader.next_test_batch(batchsz)
                test_loss, test_accuracy = sess.run( [convNet.loss_val, convNet.accuracy], feed_dict=
                                           {convNet.x:batch_imgs_test, convNet.y_:batch_labs_test, convNet.keep_prob:1.0})
                plotter.x_axis_losses_test.append(i)
                plotter.x_axis_acc_test.append(i)
                plotter.losses_test.append(test_loss)
                plotter.acc_test.append(test_accuracy)
                test_logger.log(dt.datetime.now(), test_loss, test_accuracy, i)

            
            #if i % plot_train_accuracy_every == 0:
            #    plotter.plotTrainLossAccuracy('results/')

            if i % plot_test_accuracy_every == 0:
                plotter.plotTrainVsTestAcc('results/', batchsz)
                


        timeDelta = ti.time() - time0
        print("Training time took %.2f seconds, %d minibatches,  size %d" % (timeDelta, mbatches, batchsz))

        # Measure net accuracy
        print('Measuring net accuracy on %d batches...' % measure_test_batches)
        measure_test_batches = 100
        sum_acc = 0
        for i in range(measure_test_batches):
            batch_imgs, batch_labs = mnistReader.next_test_batch(batchsz)
            #batch_imgs, batch_labs = mnistReaderOrg.test.next_batch(batchsz)
            sum_acc += sess.run( convNet.accuracy, feed_dict= 
                                            { convNet.x:batch_imgs, convNet.y_:batch_labs, convNet.keep_prob:1.0 } )
        print('Net\'s accuracy on test images is {:.2%}.'.format( sum_acc / measure_test_batches ) )

