"""This tutorial introduces the LeNet5 neural network architecture
using Theano.  LeNet5 is a convolutional neural network, good for
classifying images. This tutorial shows how to build the architecture,
and comes with all the hyper-parameters you need to reproduce the
paper's MNIST results.


This implementation simplifies the model in the following ways:

 - LeNetConvPool doesn't implement location-specific gain and bias parameters
 - LeNetConvPool doesn't implement pooling by average, it implements pooling
   by max.
 - Digit classification is implemented with a logistic regression rather than
   an RBF network
 - LeNet5 was not fully-connected convolutions at second layer

References:
 - Y. LeCun, L. Bottou, Y. Bengio and P. Haffner:
   Gradient-Based Learning Applied to Document
   Recognition, Proceedings of the IEEE, 86(11):2278-2324, November 1998.
   http://yann.lecun.com/exdb/publis/pdf/lecun-98.pdf

"""

from __future__ import print_function

import os
import sys
import timeit

import numpy

import theano
import theano.tensor as T
from theano.tensor.signal import pool
from theano.tensor.nnet import conv2d

from project_nn import LogisticRegression, load_data, HiddenLayer, HighwayLayer, myMLP, train_nn, RMSprop, Momentum,HighwayNetwork


def test_Highway(learning_rate=0.1, n_epochs=200, n_hidden=10, n_hiddenLayers=1, n_highwayLayers = 5, 
                 activation_hidden = T.nnet.nnet.relu, activation_highway = T.nnet.nnet.sigmoid,
                 dataset='mnist.pkl.gz', batch_size=500,verbose=False):
    
    """ Demonstrates lenet on MNIST dataset

    :type learning_rate: float
    :param learning_rate: learning rate used (factor for the stochastic
                          gradient)

    :type n_epochs: int
    :param n_epochs: maximal number of epochs to run the optimizer

    :type dataset: string
    :param dataset: path to the dataset used for training /testing (MNIST here)

    :type nkerns: list of ints
    :param nkerns: number of kernels on each layer
    """

    rng = numpy.random.RandomState(23455)

    datasets = load_data(dataset)

    train_set_x, train_set_y = datasets[0]
    valid_set_x, valid_set_y = datasets[1]
    test_set_x, test_set_y = datasets[2]

    # compute number of minibatches for training, validation and testing
    n_train_batches = train_set_x.get_value(borrow=True).shape[0]
    n_valid_batches = valid_set_x.get_value(borrow=True).shape[0]
    n_test_batches = test_set_x.get_value(borrow=True).shape[0]
    
    n_in = train_set_x.get_value(borrow=True).shape[1]
    
    n_train_batches //= batch_size
    n_valid_batches //= batch_size
    n_test_batches //= batch_size

    # allocate symbolic variables for the data
    index = T.lscalar()  # index to a [mini]batch

    # start-snippet-1
    x = T.matrix('x')   # the data is presented as rasterized images
    y = T.ivector('y')  # the labels are presented as 1D vector of
                        # [int] labels

    ######################
    # BUILD ACTUAL MODEL #
    ######################
    
    mlp_net = HighwayNetwork(
        rng=rng, 
        input=x,
        n_in=n_in, 
        n_hidden=n_hidden, 
        n_out=10, 
        n_hiddenLayers=n_hiddenLayers, 
        n_highwayLayers = n_highwayLayers,
        activation_hidden = activation_hidden,
        activation_highway = activation_highway
    )
    
    print('... building the model')
    
    # the cost we minimize during training is the negative log likelihood of
    # the model plus the regularization terms (L1 and L2); cost is expressed
    # here symbolically
    cost = ( mlp_net.logRegressionLayer.negative_log_likelihood(y)
        #+ L1_reg * L1
        #+ L2_reg * L2_sqr
    )
            
    # compiling a Theano function that computes the mistakes that are made
    # by the model on a minibatch    
    test_model = theano.function(
        inputs=[index],
        outputs=mlp_net.logRegressionLayer.errors(y),
        givens={
            x: test_set_x[index * batch_size:(index + 1) * batch_size],
            y: test_set_y[index * batch_size:(index + 1) * batch_size]
        }
    )

    validate_model = theano.function(
        inputs=[index],
        outputs=mlp_net.logRegressionLayer.errors(y),
        givens={
            x: valid_set_x[index * batch_size:(index + 1) * batch_size],
            y: valid_set_y[index * batch_size:(index + 1) * batch_size]
        }
    )

    # compute the gradient of cost with respect to theta (sotred in params)
    # the resulting gradients will be stored in a list gparams
    #gparams = [T.grad(cost, param) for param in mlp_net.params]

    # specify how to update the parameters of the model as a list of
    # (variable, update expression) pairs

    #updates = [
    #    (param, param - learning_rate * gparam)
    #    for param, gparam in zip(mlp_net.params, gparams)
    #]
    updates = RMSprop(cost,mlp_net.params)
    #updates = Momentum(cost, mlp_net.params, eps = learning_rate,alpha = 0.9)
    
    # compiling a Theano function `train_model` that returns the cost, but
    # in the same time updates the parameter of the model based on the rules
    # defined in `updates`
    train_model = theano.function(
        inputs=[index],
        outputs=cost,
        updates=updates,
        givens={
            x: train_set_x[index * batch_size: (index + 1) * batch_size],
            y: train_set_y[index * batch_size: (index + 1) * batch_size]
        }
    )
    
    train_nn(train_model, validate_model, test_model,
            n_train_batches, n_valid_batches, n_test_batches, n_epochs, verbose)