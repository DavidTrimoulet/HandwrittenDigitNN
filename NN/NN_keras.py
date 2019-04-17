from keras.models import Sequential
from keras.layers import Dense, Activation


class NnKeras():

    def __init__(self):
        self.model = Sequential()
        self.score = 0

    def nn_model(self, X_train, Y_train, X_test, Y_test, shape=(100,100,3, )):
        self.model.add(Dense(32, activation='relu', input_shape=shape))
        self.model.add(Dense(32, activation='relu', ))
        self.model.add(Dense(int(Y_train.shape[1]), activation='softmax'))

        self.model.compile(optimizer='adam',
                      loss='categorical_crossentropy',
                      metrics=['accuracy'])
        self.model.fit(X_train, Y_train, epochs=10, batch_size=32)

        self.score = self.model.evaluate(X_test, Y_test, batch_size=128)
        print("score:",  self.score)

    def predict(self, X):
        return self.model.predict_on_batch(X)