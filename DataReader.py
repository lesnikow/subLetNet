import numpy as np
from PIL import Image
import os, math
import csv

class DataReader(object):
	# read in training images from given image directory
	def readImages(self, imgDir) :
		row = 171
		col = 256
		images = np.zeros((len(os.listdir(imgDir)) - 1, row * col))
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
		          # convert rgb to greyscale
		          data[i * col + j] = 0.2989 * r + 0.5870 * g + 0.1140 * b
		          arr2d[i, j] = data[i * col + j]
		  images[index, :] = data[:]
		  imageIds.append(imgId)
		  index += 1
		  print(str(index) + '/' + str(len(os.listdir(imgDir)) - 1))
		return imageIds, images

	def isNumber(self, s):
	    try:
	        float(s)
	        return True
	    except ValueError:
	        return False

	# read in all labels from .csv file (including the ones not in training data)
	# ans store them in a hashmap, where key is the image id, and value is price bin
	def readLabels(self, labelPath) :
	    priceBins = {}
	    with open(labelPath, 'r') as f:
	        reader = csv.reader(f)
	        priceList = list(reader)

	    for i in range(len(priceList)) :
	        if priceList[i][0] == 'id' :
	            continue
	        id = priceList[i][0]
	        price = priceList[i][1]
	        price = price.replace('$', '')
	        if not self.isNumber(price) :
	            price = 0
	        price = float(price)
	        priceBin = -1
	        if price >= 0 and price < 70 :
	            priceBin = 0
	        elif price >= 70 and price < 150 :
	            priceBin = 1
	        elif price >= 150 and price < 225 :
	            priceBin = 2
	        else :
	            priceBin = 3
	        priceBins[id] = priceBin
	    return priceBins


