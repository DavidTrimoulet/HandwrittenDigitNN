from unittest import TestCase
from NN import NeuralNetwork
import tensorflow as tf

class TestNeuralNetworkTemplate(TestCase):

    def setUp(self):
        H = [(6, "sigmoid"), (4, "relu")]
        self.nn = NeuralNetwork.NeuralNetworkTemplate(25, 4, H)

    def test_initHiddenLayer(self):
        self.assertEqual(self.nn.layerParameters["W1"].shape,  [6 , 25])
        self.assertEqual(self.nn.layerParameters["b1"].shape, [6, 1])
        self.assertEqual(self.nn.layerParameters["W2"].shape,  [4, 6])
        self.assertEqual(self.nn.layerParameters["b2"].shape, [4, 1])
        print(self.nn.layerParameters)

    def test_initActivations(self):
        self.assertEqual(self.nn.activations[0], "sigmoid")
        self.assertEqual(self.nn.activations[1], "relu")
        print(self.nn.activations)

    def test_forwardProp(self):
        X = tf.placeholder(tf.float32, shape=[ 25 , None], name="X")
        print(self.nn.forwardPropagation(X))
