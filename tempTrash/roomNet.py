"""
Class representing a conv net model for room data set.
"""

class RoomNet():
    def __init__(self, learning_rate):
        """
        Instantiates a model of class ConvNet
        :param learning_rate: learning rate for SGD optimizer
        """
        self.learning_rate = learning_rate

        self.x = tf.placeholder(tf.float32, [None, totalPixels])
        self.y = tf.placeholder(tf.float32, [None, classes])
