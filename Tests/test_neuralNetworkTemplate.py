from unittest import TestCase
from NN import NeuralNetwork
import tensorflow as tf
from tools import Tools
from pathlib import Path

class TestNeuralNetworkTemplate(TestCase):

    def setUp(self):
        self.H = [(6, "sigmoid"), (4, "relu")]
        self.nn = NeuralNetwork.NeuralNetwork()

    def test_initHiddenLayer(self):
        parameters = self.nn.initHiddenLayer(self.H, 25, 6)
        self.assertEqual(parameters["W1"].shape,  [6 , 25])
        self.assertEqual(parameters["b1"].shape, [6, 1])
        self.assertEqual(parameters["W2"].shape,  [4, 6])
        self.assertEqual(parameters["b2"].shape, [4, 1])
        print(parameters)

    def test_initActivations(self):
        activations = self.nn.initActivations( self.H )
        self.assertEqual(activations[0], "sigmoid")
        self.assertEqual(activations[1], "relu")
        print(activations)

    def test_forwardProp(self):
        parameters = self.nn.initHiddenLayer(self.H, 25, 6)
        activations = self.nn.initActivations(self.H)
        X = tf.placeholder(tf.float32, shape=[ 25 , None], name="X")
        Z = self.nn.forward_propagation(X, parameters, activations)
        print(Z)

    def test_Cost(self):
        parameters = self.nn.initHiddenLayer(self.H, 25, 6)
        activations = self.nn.initActivations(self.H)
        X = tf.placeholder(tf.float32, shape=[25, None], name="X")
        Y = tf.placeholder(tf.float32, shape=[ 25 , None], name="X")
        cost = self.nn.compute_cost(self.nn.forward_propagation(X, parameters, activations), Y)
        print(cost)

    def test_load_hand_shown_image(self):
        p = Path('..')
        imagePath = p / "Data" / "Sign-Language-image" / "Dataset" / "0" / "IMG_1118.JPG"
        width, height, image_vectorized = Tools.get_image(imagePath)
        self.assertEqual(len(image_vectorized), 30000)
        self.assertEqual(height, 100)
        self.assertEqual(width, 100)

    def test_load_hand_shown_images(self):
        p = Path('..')
        imageFolderPath = p / "Data" / "Sign-Language-image" / "Dataset"
        m, width, height, training_set, training_label = Tools.load_hand_shown_image(imageFolderPath, True)
