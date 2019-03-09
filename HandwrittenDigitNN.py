from tools import Tools

if __name__ == '__main__':
    m, width, height, training_set = Tools.load_image("Data\\train-images.idx3-ubyte")
    training_label = Tools.load_label("Data\\train-labels.idx1-ubyte")
    m_test, width_test, height_test, test_set = Tools.load_image("Data\\t10k-images.idx3-ubyte")
    test_label = Tools.load_label("Data\\t10k-labels.idx1-ubyte")

    Tools.display_first_hundred_images(training_set)
    print(training_label)
    print("test")
