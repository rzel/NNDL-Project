
import numpy
import theano
import theano.tensor as T

# from scipy.ndimage.interpolation import shift
from PIL import Image

def translate_image(rng, input, max_shift=3):
    num_data = input.shape[0]
    old_input = input.reshape((num_data, 3, 32, 32)).transpose(0,2,3,1)    
    
    size = rng.randint(-max_shift,max_shift+1, size=(num_data,2))
    
    new_input = numpy.ndarray(shape=(num_data, 32, 32, 3),dtype=theano.config.floatX)
    
    for i in xrange(num_data):
        aux = numpy.roll(old_input[i],size[i][0],axis=1)
        aux = numpy.roll(aux,size[i][1],axis=2)
        new_input[i] = aux
 
    return (new_input.transpose(0,3,1,2)).reshape((num_data, 3*32*32))   

def rotate_image(rng, input, max_shift=10):
    num_data = input.shape[0]
    old_input = input.reshape((num_data, 3, 32, 32)).transpose(0,2,3,1)    

    size = rng.randint(-max_shift,max_shift+1, size=(num_data))

    new_input = numpy.ndarray(shape=(num_data, 32, 32, 3),dtype=theano.config.floatX)

    for i in xrange(num_data):
        img = Image.fromarray(numpy.uint8(old_input[i]*255))
        new_input[i] = numpy.asarray(img.rotate(size[i]))/255.0

    return (new_input.transpose(0,3,1,2)).reshape((num_data, 3*32*32))   


def flip_image(rng, input):
    num_data = input.shape[0]
    old_input = input.reshape((num_data, 3, 32, 32)).transpose(0,2,3,1)   
    
    new_input = numpy.ndarray(shape=(num_data, 32, 32, 3),dtype=theano.config.floatX)
    p = numpy.random.uniform(0,1,num_data)
    
    for i in xrange(num_data):
        aux = old_input[i]
        if (p[i] > 0.5):
            aux = numpy.fliplr(aux)
        new_input[i] = aux

    return (new_input.transpose(0,3,1,2)).reshape((num_data, 3*32*32))   


def zero_padding(input, input_shape, n_out_dim):
    size = (input_shape[0],n_out_dim-input_shape[1],input_shape[2],input_shape[3])
    pad = theano.shared(name='zeros',value=numpy.zeros(shape=size,dtype=theano.config.floatX)) 
    return T.concatenate([input, pad], axis=1)
