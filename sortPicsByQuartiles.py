"""
Sorts pictures in img_dir into quartiles according to the prices in path,
placing each quartile in its own directory.
"""

import csv
import os

imgs_dir = 'par11_000/'
path = 'labels/parPrices.csv'

prices = []
price_map = {}

with open(path, 'rb') as csvfile:
    reader = csv.DictReader(csvfile)
    for row in reader:
        try: 
            price_map[row['id']] = int(row['price'])
        except:
            print('None int literal %s found' % row['price'])
        try: 
            prices.append(int(row['price']))
        except:
            print('None int literal %s found' % row['price'])

prices.sort()

# Make quartiles
quart_length = len(prices) / 4

q = {}
for i in range(1, 4):
    q[i] = prices[quart_length * i]

dir_list = os.listdir(imgs_dir)

for file in dir_list:
    print(file)
    file_name = file.split('.')[0]
    try:
        #print(file_name)
        file_price = price_map[file_name]
        #print(file_price)
        if file_price < q[1]:
           print('quartile 1 found')
           os.rename(imgs_dir + file, imgs_dir + 'quart1/' + file)
        if q[1] <= file_price and file_price < q[2]:
           print('quartile 2 found')
           os.rename(imgs_dir + file, imgs_dir + 'quart2/' + file)
        if q[2] <= file_price and file_price < q[3]:
           print('quartile 3 found')
           os.rename(imgs_dir + file, imgs_dir + 'quart3/' + file)
        if q[3] <= file_price:
           print('quartile 4 found')
           os.rename(imgs_dir + file, imgs_dir + 'quart4/' + file)

    except:
        print('Error encounted.')


