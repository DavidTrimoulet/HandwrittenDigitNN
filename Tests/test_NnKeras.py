from unittest import TestCase
from NN import NN_keras
import tensorflow as tf
from tools import Tools
from pathlib import Path

class TestNnKeras(TestCase):
    def test_nn_model(self):
        X_train = tf.placeholder(name="X_train", dtype=tf.float32, shape=[30000, 2067])
        Y_train = tf.placeholder(name="Y_train", dtype=tf.float32, shape=[10, 2067])
        X_test = tf.placeholder(name="X_test", dtype=tf.float32, shape=[30000, 200])
        Y_test = tf.placeholder(name="Y_test", dtype=tf.float32, shape=[10, 200])
        nn = NN_keras.NnKeras()
        nn.nn_model(X_train, Y_train, X_test, Y_test)
