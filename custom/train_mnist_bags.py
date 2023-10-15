import numpy as np
import tensorflow as tf
import logging
tf.get_logger().setLevel(logging.ERROR)
from mil.data.datasets import mnist_bags
(bags_train, y_train, train_ins), (bags_test, y_test, test_ins) = mnist_bags.load()
# maximum  number of instances in the training set
max_len_train = np.max([len(bag) for bag in bags_train])
max_len_test = np.max([len(bag) for bag in bags_test])

max_ = np.max([max_len_train, max_len_test])
max_
bags_train_1D =[np.array(bag).reshape(-1, 28*28) for bag in bags_train]
bags_test_1D =[np.array(bag).reshape(-1, 28*28) for bag in bags_test]
