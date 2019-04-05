import numpy as np
import tensorflow as tf
from reportlab.platypus.para import paragraphEngine


class NeuralNetworkTemplate():

    def generate_placehorlders(self):
        X = tf.placeholder(tf.float32, shape= [n_x, None], name="X")
        Y = tf.placeholder(tf.float32, shape= [n_y, None], name="Y")
        return X, Y

    def train(self, optimizer):
        sess = tf.Session()
        sess.run(optimizer)
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
        parameters = {"layerParameters" : {},
                      "activations" : []}
        with tf.variable_scope("NeuralNetwork", reuse=tf.AUTO_REUSE):
            parameters["layerParameters"]["W1"] = tf.get_variable("W1", [H[0][0], n_x], initializer=tf.contrib.layers.xavier_initializer(seed=1))
            parameters["layerParameters"]["b1"] = tf.get_variable("b1", [H[0][0], 1], initializer = tf.zeros_initializer())

            for i in range(1, len(H)):
                parameters["layerParameters"]["W" + str(i+1)] = tf.get_variable("W" + str(i+1), [H[i][0] , H[i-1][0] ], initializer = tf.contrib.layers.xavier_initializer(seed = 1))
                parameters["layerParameters"]["b" + str(i+1)] = tf.get_variable("b" + str(i+1), [H[i][0],1], initializer = tf.zeros_initializer())

            parameters["layerParameters"]["W" + str(len(H)+1)] = tf.get_variable("W" + str(len(H)+1), [ n_y , H[len(H)-1 ][0] ],initializer=tf.contrib.layers.xavier_initializer(seed=1))
            parameters["layerParameters"]["b" + str(len(H)+1)] = tf.get_variable("b" + str(len(H)+1), [n_y, 1], initializer=tf.zeros_initializer())
        return parameters

    def initActivations(self, H, parameters):
        for l in H :
            parameters["activations"].append(l[1])

        return parameters

    def forwardPropagation(self, X , parameters):
        Z = tf.add(tf.matmul(parameters["layerParameters"]["W1"], X ), parameters["layerParameters"]["b1"])
        for i in range(1, len(parameters["activations"]) + 1 ):
            print("i:", i-1, ", activations:", parameters["activations"][i-1])
            A = tf.nn.relu(Z) if parameters["activations"][i-1] == "relu" else tf.nn.sigmoid(Z)
            Z = tf.add(tf.matmul(parameters["layerParameters"]["W" + str(i+1)] ,A), parameters["layerParameters"]["b" + str(i+1)] )

        return Z

    def compute_cost(self, Z, Y):
        logits = tf.transpose(Z)
        labels = tf.transpose(Y)
        cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=labels))
        return cost

    def initialize_parameters(self, H, n_x, n_y ):
        return self.initActivations( H , self.initHiddenLayer(H, n_x, n_y) )

    def model(self, X_train, Y_train, X_test, Y_test, learning_rate = 0.0001, num_epochs = 1500, minibatch_size = 32, print_cost = True, H=[(4,"sigmoid"),(4,"sigmoid"),(4,"sigmoid")]):
        (n_x, m) = X_train.shape
        n_y = Y_train.shape[0]
        costs = []
        X, Y = self.generate_placehorlders(n_x, n_y)
        parameters = self.initialize_parameters(H, n_x, n_y,)

        Z = self.forwardPropagation(X, parameters)

        cost = self.compute_cost(Z, Y)
        optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(cost)
