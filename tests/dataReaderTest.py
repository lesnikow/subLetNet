from DataReader import DataReader

# read in data
imgDir = 'par1000Sorted/train/256x171/'
labelPath = 'labels/parPrices.csv'

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

print('labels is %s' % labels)
