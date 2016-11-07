
import csv
import os

path = 'labels/parPrices.csv'

prices = []
priceMap = {}



with open(path, 'rb') as csvfile:
    reader = csv.DictReader(csvfile)
    for row in reader:
        #print ', '.join(row)
        #print(row['id'], row['price'])
        try: 
            priceMap[row['id']] = int(row['price'])
        except:
            print('None int literal %s found' % row['price'])
        
        try: 
            prices.append(int(row['price']))
        except:
            print('None int literal %s found' % row['price'])


#print('priceMap is %s' % priceMap)

prices.sort()

# Make quartiles
quart_length = len(prices) / 4

q = {}
for i in range(1, 4):
    q[i] = prices[quart_length * i]

dir_list = os.listdir('par3300')

for file in dir_list:
    print(file)
    file_name = file.split('.')[0]
    try:
        #print(file_name)
        file_price = priceMap[file_name]
        #print(file_price)
        if file_price < q[1]:
           print('quartile 1 found')
           os.rename('par3300/' + file, 'par3300/quart1/' + file)
        if q[1] <= file_price and file_price < q[2]:
           print('quartile 2 found')
           os.rename('par3300/' + file, 'par3300/quart2/' + file)
        if q[2] <= file_price and file_price < q[3]:
           print('quartile 3 found')
           os.rename('par3300/' + file, 'par3300/quart3/' + file)
        if q[3] <= file_price:
           print('quartile 4 found')
           os.rename('par3300/' + file, 'par3300/quart4/' + file)

    except:
        print('Error encounted.')


