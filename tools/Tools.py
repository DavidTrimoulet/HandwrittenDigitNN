import numpy as np
from matplotlib import pyplot as plt


def load_label(path, test=False):
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


def load_image(path, test=False):
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

def convert_from_vector_to_array(training_vector, vectorOutputNumber):
    data = np.zeros( (vectorOutputNumber, training_vector.shape[1]) )
    for i in range(0, training_vector.shape[1]):
        #print(data[training_vector[0][i]])
        #print([i])
        data[training_vector[0][i]][i] = 1

    return data

def display_n_images(data, n):
    image = np.zeros((28,28))
    for k in range(0,n):
        displayImage(data[:, k])

def displayImage(image_raw) :
    image = np.zeros((28, 28))
    image = image_raw.reshape((28, 28))
    plt.imshow(image)
    plt.show()