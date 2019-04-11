import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.python.framework import ops

class NeuralNetwork():

    def generate_placehorlders(self, n_x, n_y):
        X = tf.placeholder(tf.float32, shape= [n_x, None], name="X")
        Y = tf.placeholder(tf.float32, shape= [n_y, None], name="Y")
        return X, Y

    def cost(self, logits, labels):
        z = tf.placeholder(tf.float32, name="z")
        y = tf.placeholder(tf.float32, name="y")
        cost = tf.nn.sigmoid_cross_entropy_with_logits(logits=z,  labels=y)
        sess = tf.Session()
        cost = sess.run(cost, feed_dict={z: logits, y: labels})
        sess.close()

        return cost

    def initialize_parameters(self, H, n_x, n_y ):
        return self.initHiddenLayer(H, n_x, n_y) , self.initActivations( H )

    def initHiddenLayer(self, H, n_x, n_y):
        parameters = {}
        with tf.variable_scope("NeuralNetwork", reuse=tf.AUTO_REUSE):
            parameters["W1"] = tf.get_variable("W1", [H[0][0], n_x], initializer=tf.contrib.layers.xavier_initializer(seed=1))
            parameters["b1"] = tf.get_variable("b1", [H[0][0], 1], initializer = tf.zeros_initializer())

            for i in range(1, len(H)):
                parameters["W" + str(i+1)] = tf.get_variable("W" + str(i+1), [H[i][0] , H[i-1][0] ], initializer = tf.contrib.layers.xavier_initializer(seed = 1))
                parameters["b" + str(i+1)] = tf.get_variable("b" + str(i+1), [H[i][0],1], initializer = tf.zeros_initializer())

            parameters["W" + str(len(H)+1)] = tf.get_variable("W" + str(len(H)+1), [ n_y , H[len(H)-1 ][0] ],initializer=tf.contrib.layers.xavier_initializer(seed=1))
            parameters["b" + str(len(H)+1)] = tf.get_variable("b" + str(len(H)+1), [n_y, 1], initializer=tf.zeros_initializer())
        return parameters

    def initActivations(self, H ):
        activations = []
        for layer in H :
            activations.append(layer[1])

        return activations

    def forward_propagation(self, X , parameters, activations, rate=0.0):
        Z = tf.add(tf.matmul(parameters["W1"], X ), parameters["b1"])
        for i in range(1, len(activations) + 1 ):
            print("i:", i-1, ", activations:", activations[i-1])
            A = tf.nn.relu(Z) if activations[i-1] == "relu" else tf.nn.sigmoid(Z)
            A = tf.nn.dropout(A , rate=rate)
            Z = tf.add(tf.matmul(parameters["W" + str(i+1)] ,A), parameters["b" + str(i+1)] )
        return Z

    def compute_cost(self, Z, Y):
        logits = tf.transpose(Z)
        labels = tf.transpose(Y)
        cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=labels))
        return cost

    def model(self, X_train, Y_train, X_test, Y_test, dropout_rate=0.1, starter_learning_rate = 0.1, gradient="gradient_descent", num_epochs = 1500, print_cost = True, H=[(4,"sigmoid"),(4,"sigmoid"),(10,"sigmoid")]):

        ops.reset_default_graph()
        (n_x, m) = X_train.shape
        n_y = Y_train.shape[0]
        costs = []
        X, Y = self.generate_placehorlders(n_x, n_y)
        parameters, activations = self.initialize_parameters(H, n_x, n_y)
        Z = self.forward_propagation(X, parameters, activations, dropout_rate)

        cost = self.compute_cost(Z, Y)
        print(cost)
        if gradient == "adam":
            optimizer = tf.train.AdamOptimizer(learning_rate=starter_learning_rate).minimize(cost)
        else :
            global_step = tf.Variable(0, trainable=False)
            learning_rate = tf.train.exponential_decay(starter_learning_rate, global_step, 10000, 0.96, staircase=True)
            optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(cost)
        print(optimizer)
        init = tf.global_variables_initializer()

        with tf.Session() as sess:

            sess.run(init)

            for epoch in range(num_epochs):

                _, epoch_cost = sess.run([optimizer, cost], feed_dict={X: X_train, Y: Y_train})
                # Print the cost every epoch
                if print_cost == True and epoch % 1 == 0:
                    print("Cost after epoch %i: %f" % (epoch, epoch_cost))
                if print_cost == True and epoch % 5 == 0:
                    costs.append(epoch_cost)

            # plot the cost
            plt.plot(np.squeeze(costs))
            plt.ylabel('cost')
            plt.xlabel('iterations (per tens)')
            plt.title("Learning rate =" + str(starter_learning_rate))
            plt.show()

            # lets save the parameters in a variable
            parameters = sess.run( parameters )
            #print(sess.run(parameters))
            print("Parameters have been trained!")

            # Calculate the correct predictions
            print(tf.argmax(Z))
            print(tf.argmax(Y))
            correct_prediction = tf.equal(tf.argmax(Z), tf.argmax(Y))

            # Calculate accuracy on the test set
            accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

            print("Train Accuracy:", accuracy.eval({X: X_train, Y: Y_train}))
            print("Test Accuracy:", accuracy.eval({X: X_test, Y: Y_test}))
            self.parameters = parameters
            return parameters , activations

    def predict(self, parameters, activations , X ):
        params = {}

        for key in parameters:
            params[str(key)] = tf.convert_to_tensor( parameters[str(key)] )
        print("x shape:", X.shape)
        x = tf.placeholder("float", [X.shape[0], 1])

        z3 = self.forward_propagation(x, params, activations)

        p = tf.argmax(z3)

        sess = tf.Session()
        prediction = sess.run(p, feed_dict={x: X})
        return prediction