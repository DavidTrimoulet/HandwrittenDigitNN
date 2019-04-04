import numpy as np
from matplotlib import pyplot as plt


def load_label(path):
    data = np.array(0)
    with open(path , "rb") as f:
        magic_number = f.read(4)
        print(int.from_bytes( magic_number , byteorder='big' ))
        items = int.from_bytes( f.read(4) , byteorder='big' )
        data.resize((1, items))
        accu = 0
        byte = f.read(1)
        while byte != b"":
            data[0,accu] = int.from_bytes( byte , byteorder='big' )
            # Do stuff with byte.
            byte = f.read(1)
            accu+=1
    return data


def load_image(path):
    data = np.array(0)
    with open(path , "rb") as f:
        magic_number = f.read(4)
        print(int.from_bytes( magic_number , byteorder='big' ))
        m = int.from_bytes(f.read(4), byteorder='big')
        print(m)
        width = int.from_bytes( f.read(4), byteorder='big' )
        height = int.from_bytes( f.read(4) , byteorder='big' )
        print(width, height)
        data.resize( (width * height * m, 1 ) )
        byte = f.read(1)
        accu = 0
        while byte != b"":
            data[accu] = int.from_bytes( byte , byteorder='big' )
            accu+=1
            # Do stuff with byte.
            byte = f.read(1)
        data = data.reshape(( m , width * height)).T
        print(data.shape)
    return m, width, height, data

def display_first_hundred_images(data):
    image = np.zeros((28,28))

    for k in range(0,10):
        image = data[:, k].reshape((28,28))
        plt.imshow(image)
        plt.show()