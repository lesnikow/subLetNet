"""
subLetNet3.py

Conv net for house share images, from scratch.
"""

import numpy as np
import tensorflow as tf
import os
import ConvModel as cm

print("Starting %s!" % __file__)

# Main Training Block
if __name__ == "__main__":
    print("Starting the main training block!")

    #train_x, train_y, test_x, test_y = read(TRAIN_FILE, TEST_FILE)
    #num_train, bsz = len(train_x), FLAGS,batch_size

    # Launch tf session
    print("Launching TensorFlow Session!")
    with tf.Session() as sess:
        #Instantiate Model
        model = cm.ConvModel(10, 10e-4)

