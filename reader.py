"""
The Reader class reads in the text data for our n gram model.
"""
import numpy as np

class Reader:   
    def read(self, train_file, test_file):
        """
        Read and parse the file, building the vectorized representations of the input and output.

        :param train_file: Path to the training file.
        :param test_file: Path to the test file.
        :return: Tuple of train_x, train_y, test_x, test_y, vocab_size
        """
        vocabulary, vocab_size, train_data, test_data = {}, 0, [], []
        with open(train_file, 'r') as f:
            for line in f:
                tokens = line.split()
                train_data.extend(tokens)
                for tok in tokens:
                    if tok not in vocabulary:
                        vocabulary[tok] = vocab_size
                        vocab_size += 1
        with open(test_file, 'r') as f:
            for line in f:
                tokens = line.split()
                test_data.extend(tokens)

        # Sanity Check, make sure there are no new words in the test data.
        assert reduce(lambda x, y: x and (y in vocabulary), test_data)

        # Vectorize, and return output tuple.
        train_data = map(lambda x: vocabulary[x], train_data)
        test_data = map(lambda x: vocabulary[x], test_data)
        return np.array(train_data[:-1]), np.array(train_data[1:]), np.array(test_data[:-1]), \
            np.array(test_data[1:]), vocab_size

