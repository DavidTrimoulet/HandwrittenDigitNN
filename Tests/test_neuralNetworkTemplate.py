from unittest import TestCase
from NN import NeuralNetwork
import tensorflow as tf
from tools import Tools

class TestNeuralNetworkTemplate(TestCase):

    def setUp(self):
        self.H = [(6, "sigmoid"), (4, "relu")]
        self.nn = NeuralNetwork.NeuralNetwork()

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
        Z = self.nn.forward_propagation(X, parameters)
        test = tf.placeholder(tf.float32, shape=[6, None], name="test")
        self.assertEqual(Z.shape , test.shape )

    def test_Cost(self):
        parameters = self.nn.initActivations(self.H, self.nn.initHiddenLayer(self.H, 25, 6))
        X = tf.placeholder(tf.float32, shape=[25, None], name="X")
        Y = tf.placeholder(tf.float32, shape=[ 25 , None], name="X")
        cost = self.nn.compute_cost(self.nn.forwardPropagation(X, parameters), Y)
        print(cost)

    def test_load_hand_shown_image(self):
        width, height, image_vectorized = Tools.get_image("../Data/Sign-Language-image/Dataset/0/IMG_1118.JPG")
        self.assertEqual(len(image_vectorized), 30000)
        self.assertEqual(height, 100)
        self.assertEqual(width, 100)

    def test_load_hand_shown_images(self):
        Tools.load_hand_shown_image("../Data/Sign-Language-image/Dataset/", True)