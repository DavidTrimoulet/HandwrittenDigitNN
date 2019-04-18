from keras.models import Sequential
from keras.layers import Dense, Flatten, Dropout
from keras.utils import to_categorical
from keras.layers import Conv2D, MaxPooling2D
from keras.optimizers import Adam

class NnKeras():

    def __init__(self):
        self.model = Sequential()
        self.score = 0

    def nn_model(self, X_train, Y_train, X_test, Y_test, shape=(100,100,3)):
        Y_train = to_categorical(Y_train, num_classes=10)
        Y_test = to_categorical(Y_test, num_classes=10)
        print(Y_train.shape)
        print(Y_test.shape)
        self.model.add(Flatten())
        self.model.add(Dense(64, activation='relu', input_shape=shape))
        self.model.add(Dense(32, activation='relu', ))
        self.model.add(Dense(32, activation='relu', ))
        self.model.add(Dense(16, activation='relu', ))
        self.model.add(Dense(16, activation='relu', ))
        self.model.add(Dense(int(Y_train.shape[1]), activation='softmax'))

        self.model.compile(optimizer='adam',
                      loss='categorical_crossentropy',
                      metrics=['accuracy'])
        self.model.fit(X_train, Y_train, epochs=10, batch_size=32, shuffle=True)

        self.score = self.model.evaluate(X_test, Y_test, batch_size=128)
        print("score:",  self.score)

    def convolution_vgg(self, X_train, Y_train, X_test, Y_test):
        print(Y_train[0])
        Y_train = to_categorical(Y_train, num_classes=10)
        Y_test = to_categorical(Y_test, num_classes=10)
        print(Y_train[0])

        self.model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(100, 100, 3)))
        self.model.add(Conv2D(32, (3, 3), activation='relu'))
        self.model.add(MaxPooling2D(pool_size=(2, 2)))
        #self.model.add(Dropout(0.25))

        self.model.add(Conv2D(64, (3, 3), activation='relu'))
        self.model.add(Conv2D(64, (3, 3), activation='relu'))
        self.model.add(MaxPooling2D(pool_size=(2, 2)))
        #self.model.add(Dropout(0.25))

        self.model.add(Conv2D(128, (3, 3), activation='relu'))
        self.model.add(Conv2D(128, (3, 3), activation='relu'))
        self.model.add(MaxPooling2D(pool_size=(2, 2)))
        #self.model.add(Dropout(0.25))

        self.model.add(Conv2D(256, (3, 3), activation='relu', padding='same'))
        self.model.add(Conv2D(256, (3, 3), activation='relu', padding='same'))
        self.model.add(MaxPooling2D(pool_size=(2, 2), padding="same"))

        self.model.add(Conv2D(512, (3, 3), activation='relu', padding='same'))
        self.model.add(Conv2D(512, (3, 3), activation='relu', padding='same'))
        self.model.add(MaxPooling2D(pool_size=(2, 2), padding="same"))

        self.model.add(Flatten())
        self.model.add(Dense(4096, activation='relu'))
        self.model.add(Dense(2048, activation='relu'))

        self.model.add(Dense(10, activation='softmax'))
        adam = Adam(lr=0.0001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
        self.model.compile(loss='categorical_crossentropy', optimizer=adam)

        self.model.fit(X_train, Y_train, batch_size=32, epochs=10)
        score = self.model.evaluate(X_train, Y_train, batch_size=32)
        print("Train accuracy:", score)
        score = self.model.evaluate(X_test, Y_test, batch_size=32)
        print("Test accuracy:", score)

    def predict_one(self, X):
        X = X.reshape(1, X.shape[0], X.shape[1], X.shape[2])
        print(X.shape)
        return self.model.predict_classes(X)
