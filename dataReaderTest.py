from DataReader import DataReader

# read in data
imgDir = 'bos100/train/'
labelPath = 'labels/bosprices.csv'

# initialize
reader = DataReader()
# images
imageIds, images = reader.readImages(imgDir)
# ground truth (labels)
priceBins = reader.readLabels(labelPath)
# find label (prince bin) for 
# each data point in training dataset
labels = []
for id in imageIds :
    labels.append(priceBins[id])