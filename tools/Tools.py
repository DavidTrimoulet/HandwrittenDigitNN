import numpy as np
from matplotlib import pyplot as plt
from PIL import Image


def load_hand_written_label(path, test=False):
    data = np.array(0)
    with open(path, "rb") as f:
        m = load_hand_written_meta_data(f, test)
        data.resize((1, m))
        accu = 0
        byte = f.read(1)
        for i in range(0, m):
            data[0, accu] = int.from_bytes(byte, byteorder='big')
            # Do stuff with byte.
            byte = f.read(1)
            accu += 1
    return data


def load_hand_written_image(path, test=False):
    data = np.array(0)
    with open(path, "rb") as f:
        m = load_hand_written_meta_data(f, test)
        width = int.from_bytes(f.read(4), byteorder='big')
        height = int.from_bytes(f.read(4), byteorder='big')
        print(width, height)
        data.resize((width * height * m, 1))
        byte = f.read(1)
        accu = 0
        for i in range(0, m * width * height):
            data[accu] = int.from_bytes(byte, byteorder='big')
            accu += 1
            # Do stuff with byte.
            byte = f.read(1)
        data = data.reshape((m, width * height)).T
        print(data.shape)
    return m, width, height, data


def load_hand_written_meta_data(f, test):
    magic_number = f.read(4)
    print(int.from_bytes(magic_number, byteorder='big'))
    if not test:
        m = int.from_bytes(f.read(4), byteorder='big')
    else:
        m = 10
        f.read(4)

    return m


def load_hand_shown_image(path, test):
    training_set = []
    training_label = []
    m = 0
    width = 0
    height = 0
    for folder in path.iterdir():
        # folder ="9"
        print("folder:", folder.name)
        for file in folder.iterdir():
            training_label.append(int(folder.name))
            height, width, image = get_image(file)
            training_set.append(image)
            m += 1
        # print(m)
        if test:
            break
    training_set = np.asarray(training_set)
    training_label = np.asarray(training_label).reshape(len(training_label), 1)
    training_set, training_label = shuffle_data_set(training_set, training_label)
    return m, width, height, training_set, training_label


def get_image(filename):
    image_loaded = Image.open(filename.absolute())
    width, height = image_loaded.size
    image = np.asarray(image_loaded)
    return height, width, image


def shuffle_data_set(X, Y):
    rng_state = np.random.get_state()
    np.random.shuffle(X)
    np.random.set_state(rng_state)
    np.random.shuffle(Y)
    return X, Y


def one_hot_matrix(training_vector, vectorOutputNumber):
    data = np.zeros( (vectorOutputNumber, training_vector.shape[1]))
    for i in range(0, training_vector.shape[1]):
        data[training_vector[0][i]][i] = 1

    return data


def display_n_images(data, n, image_size, mode="grey"):
    image = np.zeros((image_size, image_size))
    for k in range(0, n):
        display_image(data[:, k], image_size, mode)


def display_image(image_raw, image_size, mode="grey"):
    if mode == "RGB":
        image = image_raw
    else:
        image = image_raw.reshape((image_size, image_size))
    plt.imshow(image)
    plt.show()
