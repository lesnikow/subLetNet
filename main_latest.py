"""
main.py
Implementation of a convolutional neural network for MNIST in tensorFlow.
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
from tensorflow.examples.tutorials.mnist import input_data

eval_train_accuracy_every = 50
plot_train_accuracy_every = 50
mbatches = int(1.0e3)
batchsz = 32

measure_test_batches = 50

#Main Training Block
if __name__ == "__main__":
    plotter = Plotter()
    print("Reading in data...")
    mnistReader = MnistReader("MNIST_data")
    mnistReaderOrg = input_data.read_data_sets("MNIST_data/", one_hot=True)
    #imageIter = ImageIterator("lon_small/square_28/")
    
    with tf.Session() as sess:
        convNet = ConvNet(1e-4, batchsz)
        sess.run(tf.initialize_all_variables())
        # Train net
        print("Training on %d minibatches of size %d..." % ( mbatches, batchsz) )
        time0 = ti.time()
        for i in range(mbatches):
            batch_imgs, batch_labs = mnistReader.next_train_batch(batchsz)
            batch_imgs_org, batch_labs_org = mnistReaderOrg.train.next_batch(batchsz)
            
            #print batch_labs
            #print batch_labs.shape
            #print batch_imgs
            #print batch_imgs.shape
            #print mnist_batch_labs
            #print mnist_batch_labs.shape
            #print batch_imgs_org
            #print batch_imgs_org.shape 

            loss, _ = sess.run([convNet.loss_val, convNet.train_op],
                                feed_dict = {convNet.x:batch_imgs, convNet.y_:batch_labs, convNet.keep_prob:0.5})
            """
            loss, _ = sess.run([convNet.loss_val, convNet.train_op],
                                feed_dict = {convNet.x:batch_imgs_org, convNet.y_:batch_labs_org, convNet.keep_prob:0.5})
            """
            plotter.x_axis_train.append(i)
            plotter.losses_train.append(loss)    

            if i % eval_train_accuracy_every == 0:
                train_accuracy = sess.run( convNet.accuracy, feed_dict=
                                           {convNet.x:batch_imgs, convNet.y_:batch_labs, convNet.keep_prob:1.0})
                #train_accuracy = sess.run( convNet.accuracy, feed_dict=
                #                            {convNet.x:batch_imgs_org, convNet.y_:batch_labs_org, convNet.keep_prob:1.0})
                
                plotter.x_axis_acc_train.append(i)
                plotter.accuracy_train.append(train_accuracy)
                now = dt.datetime.now()
                print("%d-%d-%d %2d:%2d:%2d: Step %6d: training accuracy: %.5g \t\t cross-ent loss: %.5g" % 
                        (now.month, now.day, now.year, now.hour, now.minute, now.second, i, train_accuracy, loss) )
            if i % plot_train_accuracy_every == 0:
                plotter.plotSaveTrain('results/')
        timeDelta = ti.time() - time0
        print("Training time took %.2f seconds, %d minibatches,  size %d" % (timeDelta, mbatches, batchsz))

        # Measure net accuracy
        print('Measuring net accuracy on %d batches...' % measure_test_batches)
        sum_acc = 0
        for i in range(measure_test_batches):
            batch_imgs, batch_labs = mnistReader.next_test_batch(batchsz)
            #batch_imgs, batch_labs = mnistReaderOrg.test.next_batch(batchsz)
            sum_acc += sess.run( convNet.accuracy, feed_dict= 
                                            { convNet.x:batch_imgs, convNet.y_:batch_labs, convNet.keep_prob:1.0 } )
        print('Net\'s accuracy on test images is {:.2%}.'.format( sum_acc / measure_test_batches ) )

