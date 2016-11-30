# parse inference result

# output file
outputFileName = '../result.txt'
outputFile = open(outputFileName)
lines = outputFile.readlines()

# key: image name
# value: dict{'correctLabel': label, 'probability': prob}
dict = {}

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
	imageName = line4.replace("\n", "")
	print correctLabel, probability, imageName
	dict[imageName] = {'correctLabel': correctLabel, 'probability': probability}
	i += 4
