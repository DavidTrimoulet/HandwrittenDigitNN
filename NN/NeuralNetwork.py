import numpy as np
import tensorflow as tf


class NeuralNetworkTemplate():

    def __init__(self, n_x, n_y, H=[(4,"sigmoid"),(4,"sigmoid"),(4,"sigmoid")]):
        self.layer = []
        self.trainingsetData = tf.placeholder(tf.float32, shape= [n_x, None], name="X")
        self.trainingsetLabel = tf.placeholder(tf.float32, shape= [n_y, None], name="Y")
        self.activations = self.initActivations(H)
        self.layerParameters = self.initHiddenLayer(H, n_x, n_y)

    def train(self):
        sess = tf.Session()
        sess.run(self.optimizer)
        sess.close()

    def linear(self, X, W, b):
        return tf.matmul(X, W) + b

    def sigmoid(self, X):
        return tf.sigmoid(X)

    def cost(self, logits, labels):
        z = tf.placeholder(tf.float32, name="z")
        y = tf.placeholder(tf.float32, name="y")
        cost = cost = tf.nn.sigmoid_cross_entropy_with_logits(logits=z,  labels=y)
        sess = tf.Session()
        cost = sess.run(cost, feed_dict={z: logits, y: labels})
        sess.close()

        return cost

    def initHiddenLayer(self, H, n_x, n_y):
        layerParam = {}
        with tf.variable_scope("NeuralNetwork", reuse=tf.AUTO_REUSE):
            layerParam["W1"] = tf.get_variable("W1", [H[0][0], n_x], initializer=tf.contrib.layers.xavier_initializer(seed=1))
            layerParam["b1"] = tf.get_variable("b1", [H[0][0], 1], initializer = tf.zeros_initializer())

            for i in range(1, len(H)):
                layerParam["W" + str(i+1)] = tf.get_variable("W" + str(i+1), [H[i][0] , H[i-1][0] ], initializer = tf.contrib.layers.xavier_initializer(seed = 1))
                layerParam["b" + str(i+1)] = tf.get_variable("b" + str(i+1), [H[i][0],1], initializer = tf.zeros_initializer())

            layerParam["W" + str(len(H)+1)] = tf.get_variable("W" + str(len(H)+1), [ n_y , H[len(H)-1 ][0] ],initializer=tf.contrib.layers.xavier_initializer(seed=1))
            layerParam["b" + str(len(H)+1)] = tf.get_variable("b" + str(len(H)+1), [n_y, 1], initializer=tf.zeros_initializer())
        return layerParam

    def initActivations(self, H):
        activations = []
        for l in H :
            activations.append(l[1])

        return activations

    def forwardPropagation(self, X):
        Z = tf.add(tf.matmul(self.layerParameters["W1"], X ), self.layerParameters["b1"])
        for i in range(1, len(self.activations) + 1 ):
            print("i:", i-1, ", activations:", self.activations[i-1])
            A = tf.nn.relu(Z) if self.activations[i-1] == "relu" else tf.nn.sigmoid(Z)
            Z = tf.add(tf.matmul(self.layerParameters["W" + str(i+1)] ,A), self.layerParameters["b" + str(i+1)] )

        return Z
