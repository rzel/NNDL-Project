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

import pandas as pd
import pickle

from project_nn import LogisticRegression, load_data, HiddenLayer, HighwayLayer, myMLP, HighwayNetwork, train_result, train_nn, RMSprop, Momentum, MomentumWithDecay, MomentumG


def test_Highway(datasets, learning_rate=0.1, rho = 0.9, n_epochs=200, n_hidden=10, n_hiddenLayers=1, n_highwayLayers = 5, 
                 activation_hidden = T.nnet.nnet.relu, activation_highway = T.nnet.nnet.sigmoid, b_T = -5, L1_reg = 0,
                 L2_reg = 0, batch_size=500,verbose=False,early_stopping=True):
    
    rng = numpy.random.RandomState(23455)

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
    itr = T.fscalar()  # index to an iteration

    # start-snippet-1
    x = T.matrix('x')   # the data is presented as rasterized images
    y = T.ivector('y')  # the labels are presented as 1D vector of
                        # [int] labels

    ######################
    # BUILD ACTUAL MODEL #
    ######################
    
    highway_net = HighwayNetwork(
        rng=rng, 
        input=x,
        n_in=n_in, 
        n_hidden=n_hidden, 
        n_out=10, 
        n_hiddenLayers=n_hiddenLayers, 
        n_highwayLayers = n_highwayLayers,
        activation_hidden = activation_hidden,
        activation_highway = activation_highway,
        b_T = b_T
    )
    
    print('... building the model')
    
    # the cost we minimize during training is the negative log likelihood of
    # the model plus the regularization terms (L1 and L2); cost is expressed
    # here symbolically
    cost = ( highway_net.logRegressionLayer.negative_log_likelihood(y)
        #+ L1_reg * L1
        #+ L2_reg * L2_sqr
    )
            
    # compiling a Theano function that computes the mistakes that are made
    # by the model on a minibatch    
    test_model = theano.function(
        inputs=[index],
        outputs=highway_net.logRegressionLayer.errors(y),
        givens={
            x: test_set_x[index * batch_size:(index + 1) * batch_size],
            y: test_set_y[index * batch_size:(index + 1) * batch_size]
        }
    )

    validate_model = theano.function(
        inputs=[index],
        outputs=highway_net.logRegressionLayer.errors(y),
        givens={
            x: valid_set_x[index * batch_size:(index + 1) * batch_size],
            y: valid_set_y[index * batch_size:(index + 1) * batch_size]
        }
    )

    updates = RMSprop(cost,highway_net.params,lr = learning_rate, rho = rho)
    
    # compiling a Theano function `train_model` that returns the cost, but
    # in the same time updates the parameter of the model based on the rules
    # defined in `updates`
    train_model = theano.function(
        inputs=[index,itr],
        outputs=cost,
        updates=updates,
        givens={
            x: train_set_x[index * batch_size: (index + 1) * batch_size],
            y: train_set_y[index * batch_size: (index + 1) * batch_size]
        },
        on_unused_input='ignore'
    )
        
    result = train_nn(train_model, validate_model, test_model, 
                      n_train_batches, n_valid_batches, n_test_batches, n_epochs, verbose, early_stopping)
             
    res =  pd.DataFrame([result.RunningTime, result.BestXEntropy, result.TestPerformance, result.BestValidationScore,
                         n_epochs, result.N_Epochs, activation_hidden, activation_highway, L2_reg, L1_reg,
                         batch_size, result.N_Iterations, n_hidden, n_hiddenLayers, n_highwayLayers, learning_rate, rho, result.Patience],                         
                        index=['Running time','XEntropy','Test performance','Best Validation score',
                                 'Max epochs','N epochs','Activation function - hidden', 'Activation function - highway','L2_reg parameter',
                                 'L1_reg parameter','Batch size','Iterations',
                                 'Hidden neurons per layer', 'Hidden Layers', 'Highway Layers', 'Learning rate', 'Rho','Patience']).transpose()
    
    res.to_csv('Results.csv',mode='a',index=None,header=False)
    idx = pd.read_csv('Results.csv').index.values[-1]
    
    pickle.dump(result.XEntropy,open("cross_entropy"+str(idx)+".p","wb"))
    print('Cross entropy is stored in cross_entropy'+str(idx)+'.p')             
    

def test_Highway_Momentum(datasets, learning_rate=0.1, lr_decay=0.95, momentum=0.9, n_epochs=200, n_hidden=10, n_hiddenLayers=1, n_highwayLayers = 5, 
                 activation_hidden = T.nnet.nnet.relu, activation_highway = T.nnet.nnet.sigmoid, b_T = -5, L1_reg = 0,
                 L2_reg = 0, batch_size=500,verbose=False, early_stopping=True):
    
    rng = numpy.random.RandomState(23455)

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
    itr = T.fscalar()  # index to an iteration

    # start-snippet-1
    x = T.matrix('x')   # the data is presented as rasterized images
    y = T.ivector('y')  # the labels are presented as 1D vector of
                        # [int] labels

    ######################
    # BUILD ACTUAL MODEL #
    ######################
    
    highway_net = HighwayNetwork(
        rng=rng, 
        input=x,
        n_in=n_in, 
        n_hidden=n_hidden, 
        n_out=10, 
        n_hiddenLayers=n_hiddenLayers, 
        n_highwayLayers = n_highwayLayers,
        activation_hidden = activation_hidden,
        activation_highway = activation_highway,
        b_T = b_T
    )
    
    print('... building the model')
    
    # the cost we minimize during training is the negative log likelihood of
    # the model plus the regularization terms (L1 and L2); cost is expressed
    # here symbolically
    cost = ( highway_net.logRegressionLayer.negative_log_likelihood(y)
        #+ L1_reg * L1
        #+ L2_reg * L2_sqr
    )
            
    # compiling a Theano function that computes the mistakes that are made
    # by the model on a minibatch    
    test_model = theano.function(
        inputs=[index],
        outputs=highway_net.logRegressionLayer.errors(y),
        givens={
            x: test_set_x[index * batch_size:(index + 1) * batch_size],
            y: test_set_y[index * batch_size:(index + 1) * batch_size]
        }
    )

    validate_model = theano.function(
        inputs=[index],
        outputs=highway_net.logRegressionLayer.errors(y),
        givens={
            x: valid_set_x[index * batch_size:(index + 1) * batch_size],
            y: valid_set_y[index * batch_size:(index + 1) * batch_size]
        }
    )

    updates = MomentumG(cost, highway_net.params, itr, lr_base=learning_rate, 
                                lr_decay=lr_decay, momentum=momentum)
    
    # compiling a Theano function `train_model` that returns the cost, but
    # in the same time updates the parameter of the model based on the rules
    # defined in `updates`
    train_model = theano.function(
        inputs=[index,itr],
        outputs=cost,
        updates=updates,
        givens={
            x: train_set_x[index * batch_size: (index + 1) * batch_size],
            y: train_set_y[index * batch_size: (index + 1) * batch_size]
        },
        on_unused_input='ignore'
    )
        
    result = train_nn(train_model, validate_model, test_model, 
                      n_train_batches, n_valid_batches, n_test_batches, n_epochs, verbose, early_stopping)
             
    res =  pd.DataFrame([result.RunningTime, result.BestXEntropy, result.TestPerformance, result.BestValidationScore,
                         n_epochs, result.N_Epochs, activation_hidden, activation_highway, L2_reg, L1_reg,
                         batch_size, result.N_Iterations, n_hidden, n_hiddenLayers, n_highwayLayers, learning_rate, lr_decay, momentum, result.Patience],                         
                        index=['Running time','XEntropy','Test performance','Best Validation score',
                                 'Max epochs','N epochs','Activation function - hidden', 'Activation function - highway','L2_reg parameter',
                                 'L1_reg parameter','Batch size','Iterations', 'Hidden neurons per layer', 'Hidden Layers', 'Highway Layers', 
                                 'Learning rate', 'lr_decay', 'momentum', 'Patience']).transpose()
    
    res.to_csv('Results.csv',mode='a',index=None,header=False)
    idx = pd.read_csv('Results.csv').index.values[-1]
    
    pickle.dump(result.XEntropy,open("cross_entropy"+str(idx)+".p","wb"))
    print('Cross entropy is stored in cross_entropy'+str(idx)+'.p') 
