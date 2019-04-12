import numpy as np
from matplotlib import pyplot as plt
import os
from PIL import Image

def load_hand_written_label(path, test=False):
    data = np.array(0)
    with open(path , "rb") as f:
        magic_number = f.read(4)
        print(int.from_bytes( magic_number , byteorder='big' ))
        if not test:
            items = int.from_bytes(f.read(4), byteorder='big')
        else :
            items = 10
            f.read(4)
        data.resize((1, items))
        accu = 0
        byte = f.read(1)
        for i in range(0,items):
            data[0,accu] = int.from_bytes( byte , byteorder='big' )
            # Do stuff with byte.
            byte = f.read(1)
            accu+=1
    return data

def load_hand_written_image(path, test=False):
    data = np.array(0)
    with open(path , "rb") as f:
        magic_number = f.read(4)
        print(int.from_bytes( magic_number , byteorder='big' ))
        if not test:
            m = int.from_bytes(f.read(4), byteorder='big')
        else :
            m = 10
            f.read(4)
        print(m)
        width = int.from_bytes( f.read(4), byteorder='big' )
        height = int.from_bytes( f.read(4) , byteorder='big' )
        print(width, height)
        data.resize( (width * height * m, 1 ) )
        byte = f.read(1)
        accu = 0
        for i in range(0 , m * width * height) :
            data[accu] = int.from_bytes( byte , byteorder='big' )
            accu+=1
            # Do stuff with byte.
            byte = f.read(1)
        data = data.reshape(( m , width * height)).T
        print(data.shape)
    return m, width, height, data

def load_hand_shown_image(path, test):
    training_set = []
    training_label = []
    m = 0
    width = 0
    height = 0
    folders = os.listdir(path)
    print(folders)
    for folder in folders :
    #folder ="9"
        files = os.listdir(path + "/" + folder)
        print("folder:", folder)
        for file in files:
            filename = path + "/" + folder + "/" + file
            training_label.append(int(folder))
            height, width , image_vectorized = get_image(filename)
            training_set = training_set + image_vectorized
            m += 1
        #print(m)
    training_set = np.asarray(training_set).reshape(( m , width * height * 3)).T
    print(training_set.shape)

    return m, width, height, training_set, training_label


def get_image(filename):
    image = Image.open(filename)
    width, height = image.size
    # if width != 100 :
    #    print("file:", file, width, height)
    pixels = image.load()
    r = []
    g = []
    b = []
    for i in range(0, width):
        for j in range(0, height):
            pixel = pixels[i, j]
            r.append(pixel[0])
            g.append(pixel[1])
            b.append(pixel[2])
    image_vectorized = r + g + b
    return height, width, image_vectorized


def convert_from_vector_to_array(training_vector, vectorOutputNumber):
    data = np.zeros( (vectorOutputNumber, training_vector.shape[1]) )
    for i in range(0, training_vector.shape[1]):
        #print(data[training_vector[0][i]])
        #print([i])
        data[training_vector[0][i]][i] = 1

    return data

def display_n_images(data, n, image_size):
    image = np.zeros((image_size,image_size))
    for k in range(0,n):
        displayImage(data[:, k], image_size)

def displayImage(image_raw, image_size) :
    image = image_raw.reshape((image_size, image_size))
    plt.imshow(image)
    plt.show()