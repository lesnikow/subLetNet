# parse inference result
from shutil import copyfile
import os
# output file
outputFileName = './results/nov30.results'
outputFile = open(outputFileName)
lines = outputFile.readlines()

# where to put images after filtering
destDir = './inference_result/'

# key: image name
# value: dict{'correctLabel': label, 'probability': prob}
dict = {}


# clear current images in output folders
dir = destDir + 'indoor'
for f in os.listdir(dir) :
	os.remove(dir + '/' + f)
dir = destDir + 'outdoor'
for f in os.listdir(dir) :
	os.remove(dir  + '/' + f)

# read through the output file
# every item takes up 4 rows
# row1: err msg
# row2: dominant label and prob
# row3: the other label and prob
# row4: image path
i = 0
while i + 3 < len(lines) :
	line2 = lines[i + 1]
	line3 = lines[i + 2]
	line4 = lines[i + 3]

	line2words = line2.split(' ')
	correctLabel = line2words[2]
	probability = line2words[4].replace("\n", "")
	probability = float(probability)
	if correctLabel == 'outdoor' and probability < 0.8 :
		correctLabel = 'indoor'
		probability = 1 - probability
	imagePath = line4.replace("\n", "")
	imageName = imagePath.split("/")[-1]
	print correctLabel, probability, imageName
	dict[imagePath] = {'correctLabel': correctLabel, 'probability': probability}
	copyfile(imagePath, destDir + correctLabel + '/' + imageName)
	i += 4
