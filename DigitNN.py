from tools import Tools
import cv2
from NN import NeuralNetwork as NN
from NN import NN_keras as NN_K
from pathlib import Path
from keras.models import load_model

ROOT_PATH = Path('.')
cap = cv2.VideoCapture(0)


def hand_written_digit():
    test = False
    datasetPath = ROOT_PATH / "Data" / "train-images.idx3-ubyte"
    m, width, height, training_set = Tools.load_hand_written_image(datasetPath, test)
    datasetLabelPath = ROOT_PATH / "Data" / "train-labels.idx1-ubyte"
    training_label = Tools.load_hand_written_label(datasetLabelPath, test)
    training_set = training_set / 255
    print(training_label)
    training_label_one_hot = Tools.one_hot_matrix(training_label, 10)
    testsetPath = ROOT_PATH / "Data" / "t10k-images.idx3-ubyte"
    m_test, width_test, height_test, test_set = Tools.load_hand_written_image(testsetPath, test)
    test_set = test_set / 255
    testsetLabelPath = ROOT_PATH / "Data" / "t10k-labels.idx1-ubyte"
    test_label = Tools.load_hand_written_label(testsetLabelPath, test)
    test_label_one_hot = Tools.one_hot_matrix(test_label, 10)
    channel = 1
    Tools.display_n_images(training_set, 10, width)

    my_network = NN.NeuralNetwork()
    # Train 0.93 test 0.93 my_network.model( training_set, training_label, test_set, test_label, learning_rate = 0.1, num_epochs = 500, H=[(50,"relu"),(25,"relu"),(12,"relu")])
    # bad my_network.model(training_set, training_label, test_set, test_label, learning_rate=0.01, num_epochs=2000, H=[(50, "relu"), (25, "relu"), (12, "relu")])
    # bad my_network.model(training_set, training_label, test_set, test_label, learning_rate=0.01, num_epochs=2000, H=[(50, "sigmoid"), (25, "sigmoid"), (12, "sigmoid")])
    # Train 0.99 test 0.97 my_network.model(training_set, training_label, test_set, test_label, learning_rate=0.1, num_epochs=2000, H=[(370, "relu"),(150, "relu"),(10, "relu"),(50, "relu"), (25, "relu"), (12, "relu")])
    # Train 0.94 test 0.94 my_network.model(training_set, training_label, test_set, test_label, learning_rate=0.08, num_epochs=500, H=[(370, "relu"),(150, "relu"),(10, "relu"),(50, "relu"), (25, "relu"), (12, "relu")])
    # Train 0.93 test 0.93 my_network.model(training_set, training_label, test_set, test_label, learning_rate=0.06, num_epochs=500, H=[(370, "relu"), (150, "relu"), (10, "relu"), (50, "relu"), (25, "relu"), (12, "relu")])
    # Train 0.95 test 0.95 my_network.model(training_set, training_label, test_set, test_label, starter_learning_rate=0.1, num_epochs=500, H=[(370, "relu"), (150, "relu"), (10, "relu"), (50, "relu"), (25, "relu"), (12, "relu")])
    # Aborted bad my_network.model(training_set, training_label, test_set, test_label, gradient="adam", starter_learning_rate=0.1, num_epochs=500, H=[(370, "relu"), (150, "relu"), (10, "relu"), (50, "relu"), (25, "relu"), (12, "relu")])
    # Train 0.95 test 0.94 my_network.model(training_set, training_label, test_set, test_label, gradient="adam", starter_learning_rate=0.0001, num_epochs=500, H=[(370, "relu"), (150, "relu"), (10, "relu"), (50, "relu"), (25, "relu"), (12, "relu")])
    # Train 0.99995 test 0.9715 my_network.model(training_set, training_label, test_set, test_label, gradient="adam", starter_learning_rate=0.001, num_epochs=500, H=[(370, "relu"), (150, "relu"), (10, "relu"), (50, "relu"), (25, "relu"), (12, "relu")])
    # Train 0.99995 test 0.9731 my_network.model(training_set, training_label, test_set, test_label, gradient="adam", starter_learning_rate=0.001, num_epochs=500, H=[(150, "relu"), (50, "relu"), (25, "relu"), (12, "relu")])
    # Train 0.98145 test 0.9618 my_network.model(training_set, training_label, test_set, test_label, gradient="adam", starter_learning_rate=0.001, num_epochs=500, H=[(50, "relu"), (25, "relu"), (12, "relu")])
    # Train 1.0 test 0.9721 my_network.model(training_set, training_label, test_set, test_label, gradient="adam", starter_learning_rate=0.01, num_epochs=500, H=[(150, "relu"), (50, "relu"), (25, "relu"), (12, "relu")])
    # Train 0.99983335 test 0.9613 my_network.model(training_set, training_label, test_set, test_label, gradient="adam", starter_learning_rate=0.01, num_epochs=500, H=[ (50, "relu"), (25, "relu"), (12, "relu")])
    # Train 1.0 test 0.971 my_network.model(training_set, training_label, test_set, test_label, gradient="adam", starter_learning_rate=0.01, num_epochs=500, H=[(150, "relu"), (50, "relu"), (25, "relu")])
    # Train 0.9996833 test 0.9637 my_network.model(training_set, training_label, test_set, test_label, gradient="adam", starter_learning_rate=0.01, num_epochs=500, H=[ (50, "relu"), (25, "relu")])
    # Train 0.9996833 test 0.9637 my_network.model(training_set, training_label, test_set, test_label, gradient="adam", starter_learning_rate=0.01, num_epochs=500, H=[(150, "relu"), (50, "relu"), (25, "relu"), (25, "relu"), (25, "relu"), (12, "relu")])
    # Train 1.0 test 0.9737 my_network.model(training_set, training_label, test_set, test_label, gradient="adam", starter_learning_rate=0.01, num_epochs=500, H=[(150, "relu"), (50, "relu"), (50, "relu"), (25, "relu") ])
    # Train 1.0 test 0.9741 my_network.model(training_set, training_label, test_set, test_label, gradient="adam", starter_learning_rate=0.01, num_epochs=500, H=[(150, "relu"), (50, "relu"), (50, "relu")])
    # Train 1.0 test 0.9725 my_network.model(training_set, training_label, test_set, test_label, gradient="adam", starter_learning_rate=0.01, num_epochs=500, H=[(150, "relu"), (50, "relu")])
    # Train 1.0 test 0.9664 my_network.model(training_set, training_label, test_set, test_label, gradient="adam", starter_learning_rate=0.01, num_epochs=500, H=[(50, "relu"), (50, "relu") , (50, "relu")])
    # Train 1.0 test 0.9664 my_network.model(training_set, training_label, test_set, test_label, gradient="adam", starter_learning_rate=0.01, num_epochs=500, H=[(150, "relu"), (150, "relu"), (50, "relu")])
    # adding dropout
    # Train 0.9980 test 0.9749 my_network.model(training_set, training_label, test_set, test_label, gradient="adam", starter_learning_rate=0.01, num_epochs=500, H=[(150, "relu"), (50, "relu"), (50, "relu")])
    # Train 0.99825 test 0.9753 my_network.model(training_set, training_label, test_set, test_label, gradient="adam", starter_learning_rate=0.01, num_epochs=500, H=[(150, "relu"), (50, "relu"), (50, "relu") , (50, "relu")])
    # Train 0.9985833 test 0.9727 my_network.model(training_set, training_label, test_set, test_label, gradient="adam", starter_learning_rate=0.01, num_epochs=500, H=[(150, "relu"), (50, "relu"), (50, "relu"), (50, "relu"), (50, "relu"), (50, "relu")])
    # Train 0.9981667 test 0.9731 my_network.model(training_set, training_label, test_set, test_label, gradient="adam", starter_learning_rate=0.01, num_epochs=500, H=[(150, "relu"), (150, "relu"), (150, "relu"), (150, "relu")])
    # Train 0.99481666 test 0.9676 my_network.model(training_set, training_label, test_set, test_label, gradient="adam", starter_learning_rate=0.01, num_epochs=500, H=[(370, "relu"), (150, "relu"), (50, "relu"), (25, "relu"), (12, "relu")])
    # Train 0.995633 test 0.9663 my_network.model(training_set, training_label, test_set, test_label, gradient="adam", starter_learning_rate=0.01, dropout_rate=0.2,num_epochs=500, H=[(150, "relu"), (50, "relu"), (50, "relu"), (50, "relu")])

    parameters, activations = my_network.model(training_set, training_label_one_hot, test_set, test_label_one_hot,
                                               gradient="adam", starter_learning_rate=0.01, num_epochs=500,
                                               H=[(150, "relu"), (150, "relu"), (50, "relu")])

    while True:
        print("get a picture number between 0 and ", test_set.shape[1], ":")
        try:
            image_number = int(input('Image:'))
        except ValueError:
            print ("Not a number")
        print(test_set)
        Tools.display_image(test_set[:, image_number], width, mode="RGB")
        predicted_number = my_network.predict(parameters, activations, test_set[:, image_number].reshape(width * width * channel, 1))
        print(test_label)
        print("it's a ", predicted_number, "should be a ", test_label[0][image_number])


def hand_shown_digit():
    test = False
    dataset_path = ROOT_PATH / "Data" / "Sign-Language-image" / "Dataset"
    save_path = ROOT_PATH / "save" / "hand_shown_digit.h5"
    my_file = Path(save_path)

    if my_file.is_file():
        my_network = NN_K.NnKeras()
        my_network.model = load_model(str(save_path))
    else:
        m, width, height, training_set, training_label = Tools.load_hand_shown_image(dataset_path, test)
        index = ((m // 4) * 3)
        test_set = training_set[index:, :, :, :]
        test_label = training_label[index:, :]
        training_set = training_set[:index, :, :, :]
        training_label = training_label[:index, :]
        Tools.display_image(training_set[55, :, :, :], 100, mode="RGB")
        my_network = NN_K.NnKeras()
        my_network.convolution_vgg(training_set, training_label, test_set, test_label)
        my_network.model.save(str(save_path))

    while True:
        et, frame = cap.read()
        img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (100, 100))

        predicted_number = my_network.predict_one(img)

        print("it's a ", predicted_number)
        cv2.imshow('frame', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()



if __name__ == '__main__':
    #hand_written_digit()
    hand_shown_digit()

