
from mnistReader import MnistReader
mnistReader = MnistReader("data/")
#print mnistReader.path


batch_imgs, batch_labs = mnistReader.next_train_batch(32)
#print batch_imgs[:1]
#print batch_labs[:1]

#batch_imgs, batch_labs = mnistReader.next_train_batch(32)
#print batch_imgs[:1]
#print batch_labs[:1]

#batch_imgs_test, batch_labs_test = mnistReader.next_test_batch(32)
#print batch_imgs_test
#print batch_labs_test

#print batch_labs_test

#oneHot = mnistReader.oneHot(batch_labs_test, 10)

#print oneHot:
