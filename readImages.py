import tensorflow as tf
import numpy as np
from PIL import Image
import os

MAX_IMAGES = 2

os.chdir("bos100Big")
filenames = os.listdir(".")[:MAX_IMAGES]  

filename_queue = tf.train.string_input_producer(filenames)

reader = tf.WholeFileReader()
key, value = reader.read(filename_queue)

images = tf.image.decode_jpeg(value, channels=1)  # 1 for greyscale, 3 for RGB

init_op = tf.initialize_all_variables()
with tf.Session() as sess:
	sess.run(init_op)

	coord = tf.train.Coordinator()
	threads = tf.train.start_queue_runners(coord=coord)

	for i in range(len(filenames)):
		image = images.eval()	#the tensorflow image
		print(image.shape)
		print image
		npArray = np.asarray(image)
		npArray = npArray[:, :, 0]  # drop from 3 dimensions to 2
		imageImage = Image.fromarray(npArray, 'L')
		imageImage.show()

	coord.request_stop()
	coord.join(threads)

