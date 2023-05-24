import numpy as np
import torch as tc
import scipy.io as sio
from pathlib import Path


def load_mnist(name):
    """
    Function loads MNIST data set. Assumes that the data file is contained in the working directory
    :param name: string, name of file including extension
    :return: data: MNIST data set.
    """
    working_dir = Path.cwd()
    get_file = working_dir / name
    data = sio.loadmat(get_file)

    return data


def int_to_one_hot(array):
    """
    converts integer array, where array includes all int up to certain value, to one-hot encoding
    :param array: 1d array of integers
    :return:
    """
    if (array.dtype != 'int') or (len(array.shape) != 1):
        print('unable to convert to one-hot because of input array formatting')
    else:
        one_hot = np.zeros((len(array), np.max(array) + 1))
        one_hot[np.arange(len(array)), array] = 1

        return one_hot


def remove_unused_one_hots(array):
    """
    removes unused columns in a 1-hot array converted from integers via the above function
    :param array: 2d array with 1-hot vectors in each row
    :return:
    """

    del_index = []
    for idx, col in enumerate(array.T):

        if not np.any(col):
            del_index.append(idx)

    return np.delete(array, del_index, 1)


def setup_mnist(data, index_list=None):
    """
    sets up mnist dataset, for supervised learning task, from single file; organizes and rescales it.
    :param data: file containing MNIST
    :param index_list: list of integers in (0,...,9) selecting which numbers to return
    :return: numpy arrays: X_train [60000, 784], Y_train [60000, 1], X_test [10000, 784], Y_test [10000, 1]
    """

    if index_list is None:
        index_list = np.arange(10)

    to_plot = []
    for key, value in data.items():                         # Rescale
        if key[0] == 't':
            data[key] = data[key]/255.0
            if key[1] == 'r':
                to_plot.append(value[0].reshape(28, 28))

    x_train = data['train' + str(index_list[0])]                              # Initialize training variables
    for i in index_list[1:]:
        x_train = np.concatenate((x_train, data['train' + str(i)]), axis=0)
    x_train = x_train.transpose()
    y_train = []
    for i in index_list:
        y_train.extend(np.ones(data['train' + str(i)].shape[0], dtype=int)*i)
    y_train = np.asarray(y_train)

    x_test = data['test' + str(index_list[0])]                                # Initialize testing variables
    for i in index_list[1:]:
        x_test = np.concatenate((x_test, data['test' + str(i)]), axis=0)
    x_test = x_test.transpose()
    y_test = []
    for i in index_list:
        y_test.extend(np.ones(data['test' + str(i)].shape[0], dtype=int)*i)
    y_test = np.asarray(y_test)

    return x_train.T, y_train, x_test.T, y_test


def mnist_for_class(index_list=None, return_torch=True, thresh=None, one_hots=False):
    """
    loads and sets up mnist dataset for classification. Uses 1-hot encoding of digits
    :param index_list: a list of indices on (0,...,9) selecting which numbers to return
    :param return_torch: whether to return arrays as numpy or torch
    :param thresh: threshold for converting pixels to binary. If none, use nonbinary pixels
    :param one_hots: bool. Whether ys returned are one hots or integers
    :return: arrays: (60000 x 794) training set, (10000 x 794) test set -- can be diff sizes depending on index_list
    """

    MNIST = load_mnist('mnist_all.mat')
    tr, ytr, ts, yts = setup_mnist(MNIST, index_list=index_list)

    if thresh:
        tr = (tr > thresh) * 1

    if one_hots:
        ytr = remove_unused_one_hots(int_to_one_hot(ytr))
        yts = remove_unused_one_hots(int_to_one_hot(yts))

    # tr = np.concatenate((tr, ytr), axis=1)
    # ts = np.concatenate((ts, yts), axis=1)

    if return_torch:
        tr = tc.from_numpy(tr)
        ts = tc.from_numpy(ts)
        ytr = tc.from_numpy(ytr)
        yts = tc.from_numpy(yts)

    return tr, ytr, ts, yts


def mnist_for_gen(index_list=None, return_torch=True, thresh=None):
    """
    loads and sets up mnist dataset for generative modelling. Uses 1-hot encoding of digits
    :param index_list: a list of indices on (0,...,9) selecting which numbers to return
    :param return_torch: whether to return arrays as numpy or torch
    :param thresh: threshold for converting pixels to binary. If none, use nonbinary pixels
    :return: arrays: (60000 x 794) training set, (10000 x 794) test set -- can be diff sizes depending on index_list
    """

    MNIST = load_mnist('mnist_all.mat')
    tr, ytr, ts, yts = setup_mnist(MNIST, index_list=index_list)

    if thresh:
        tr = (tr > thresh) * 1

    ytr = remove_unused_one_hots(int_to_one_hot(ytr))
    yts = remove_unused_one_hots(int_to_one_hot(yts))

    tr = np.concatenate((tr, ytr), axis=1)
    ts = np.concatenate((ts, yts), axis=1)

    if return_torch:
        tr = tc.from_numpy(tr)
        ts = tc.from_numpy(ts)

    return tr, ts

if __name__ == "__main__":

    # Example code for supervised learning task without 1-hot digit encoding
    DATA = load_mnist('mnist_all.mat')
    nums = [1, 2, 9]
    x_tr, y_tr, x_ts, y_ts = setup_mnist(DATA, index_list=nums)

    # Example code for generative task with 1-hot digit encoding
    train, test = mnist_for_gen(index_list=nums)

    # Example code for generative task with 1-hot digit encoding and binary pixels
    train_b, test_b = mnist_for_gen(thresh=0.8)

    #%%


