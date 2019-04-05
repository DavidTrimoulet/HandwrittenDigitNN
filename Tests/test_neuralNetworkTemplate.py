from unittest import TestCase
from NN import NeuralNetwork
import tensorflow as tf

class TestNeuralNetworkTemplate(TestCase):

    def setUp(self):
        self.H = [(6, "sigmoid"), (4, "relu")]
        self.nn = NeuralNetwork.NeuralNetworkTemplate()

    def test_initHiddenLayer(self):
        parameters = self.nn.initHiddenLayer(self.H, 25, 6)
        self.assertEqual(parameters["layerParameters"]["W1"].shape,  [6 , 25])
        self.assertEqual(parameters["layerParameters"]["b1"].shape, [6, 1])
        self.assertEqual(parameters["layerParameters"]["W2"].shape,  [4, 6])
        self.assertEqual(parameters["layerParameters"]["b2"].shape, [4, 1])
        print(parameters["layerParameters"])

    def test_initActivations(self):
        parameters = self.nn.initActivations( self.H, self.nn.initHiddenLayer(self.H, 25, 6) )
        self.assertEqual(parameters["activations"][0], "sigmoid")
        self.assertEqual(parameters["activations"][1], "relu")
        print(parameters["activations"])

    def test_forwardProp(self):
        parameters = self.nn.initActivations(self.H, self.nn.initHiddenLayer(self.H, 25, 6))
        X = tf.placeholder(tf.float32, shape=[ 25 , None], name="X")
        Z = self.nn.forwardPropagation(X, parameters)
        self.assertEqual(Z.shape , [4, None] )

    def test_Cost(self):
        parameters = self.nn.initActivations(self.H, self.nn.initHiddenLayer(self.H, 25, 6))
        X = tf.placeholder(tf.float32, shape=[25, None], name="X")
        Y = tf.placeholder(tf.float32, shape=[ 25 , None], name="X")
        cost = self.nn.compute_cost(self.nn.forwardPropagation(X, parameters), Y)
        print(cost)