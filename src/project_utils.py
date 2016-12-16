import numpy
import theano

from scipy.ndimage.interpolation import shift
from PIL import Image

def translate_image(rng, input, max_shift=3):
    num_data = input.shape[0]
    old_input = input.reshape((num_data, 3, 32, 32)).transpose(0,2,3,1)    

    size = rng.randint(-max_shift,max_shift+1, size=(num_data,2))
    
    new_input = numpy.ndarray(shape=(num_data, 32, 32, 3),dtype=theano.config.floatX)
    
    for i in range(num_data):
        shiftVector = (size[i][0],size[i][1],0)
        new_input[i] = shift(old_input[i],shift=shiftVector)
    
    return (new_input.transpose(0,3,1,2)).reshape((num_data, 3*32*32))   


def rotate_image(rng, input, max_shift=10):
    num_data = input.shape[0]
    old_input = input.reshape((num_data, 3, 32, 32)).transpose(0,2,3,1)    

    size = rng.randint(-max_shift,max_shift+1, size=(num_data))

    new_input = numpy.ndarray(shape=(num_data, 32, 32, 3),dtype=theano.config.floatX)

    for i in range(num_data):
        img = Image.fromarray(numpy.uint8(old_input[i]*255))
        new_input[i] = numpy.asarray(img.rotate(size[i]))/255.0

    return (new_input.transpose(0,3,1,2)).reshape((num_data, 3*32*32))   


def flip_image(rng, input):
    num_data = input.shape[0]
    old_input = input.reshape((num_data, 3, 32, 32)).transpose(0,2,3,1)    

    flag = rng.randint(0,2, size=(num_data))

    new_input = numpy.ndarray(shape=(num_data, 32, 32, 3),dtype=theano.config.floatX)

    for i in range(num_data):
        if flag[i]==1:
            img = Image.fromarray(numpy.uint8(old_input[i]*255))
            new_input[i] = numpy.asarray(img.transpose(Image.FLIP_LEFT_RIGHT))/255.0     
        else:
            new_input[i] = old_input[i] 

    return (new_input.transpose(0,3,1,2)).reshape((num_data, 3*32*32))   

