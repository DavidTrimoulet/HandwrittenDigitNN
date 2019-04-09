from tools import Tools
from NN import NeuralNetwork as NN

if __name__ == '__main__':
    m, width, height, training_set = Tools.load_image("Data/train-images.idx3-ubyte")
    training_label = Tools.load_label("Data/train-labels.idx1-ubyte")
    training_set = training_set / 255
    print(training_label)
    training_label = Tools.convert_from_vector_to_array(training_label, 10)
    m_test, width_test, height_test, test_set = Tools.load_image("Data/t10k-images.idx3-ubyte")
    test_set = test_set / 255
    test_label = Tools.load_label("Data/t10k-labels.idx1-ubyte")
    test_label = Tools.convert_from_vector_to_array(test_label, 10)

    print(test_label.shape)
    Tools.display_first_hundred_images(training_set)
    #print(training_label)

    my_network = NN.NeuralNetworkTemplate()
    #my_network.model( training_set, training_label, test_set, test_label, learning_rate = 0.01, num_epochs = 2000, H=[(25,"relu"),(25,"relu"),(25,"relu"),(25,"relu"),(25,"relu"),(25,"relu"),(12,"relu")])
    #my_network.model( training_set, training_label, test_set, test_label, learning_rate = 0.01, num_epochs = 500, H=[(25,"relu"),(25,"relu"),(25,"relu"),(25,"relu"),(25,"relu"),(25,"relu"),(12,"relu")])
    #my_network.model( training_set, training_label, test_set, test_label, learning_rate = 0.01, num_epochs = 500, H=[(50,"relu"),(25,"relu"),(12,"relu")])
    #93% my_network.model( training_set, training_label, test_set, test_label, learning_rate = 0.1, num_epochs = 500, H=[(50,"relu"),(25,"relu"),(12,"relu")])
    #bad my_network.model(training_set, training_label, test_set, test_label, learning_rate=0.01, num_epochs=2000, H=[(50, "relu"), (25, "relu"), (12, "relu")])
    #bad my_network.model(training_set, training_label, test_set, test_label, learning_rate=0.01, num_epochs=2000, H=[(50, "sigmoid"), (25, "sigmoid"), (12, "sigmoid")])
    # Train 0.99 test 0.97 my_network.model(training_set, training_label, test_set, test_label, learning_rate=0.1, num_epochs=2000, H=[(370, "relu"),(150, "relu"),(10, "relu"),(50, "relu"), (25, "relu"), (12, "relu")])
    my_network.model(training_set, training_label, test_set, test_label, learning_rate=0.08, num_epochs=500, H=[(370, "relu"),(150, "relu"),(10, "relu"),(50, "relu"), (25, "relu"), (12, "relu")])

