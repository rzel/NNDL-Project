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

from project_nn_resnet import LogisticRegression, load_data, HiddenLayer, ConvLayer, ConvHighwayLayer, HighwayLayer, DropoutLayer, PoolingLayer, ActivationLayer, myMLP, HighwayNetwork, train_result, train_nn_NoValidation, MomentumWithMultiStepDecay
from project_utils import translate_image, rotate_image, flip_image, zero_padding

class ResNet(object):
    def __init__(self, rng, x, n, drop_rate, training_enabled, batch_size):

        # Reshape matrix of rasterized images of shape (batch_size, 32 * 32) to a 4D tensor
        input_layer = x.reshape((batch_size, 3, 32, 32))

        self.allLayers = []
        self.allLayers.append(
            DropoutLayer(is_train=training_enabled, 
                         input=input_layer, 
                         p=1-drop_rate))

        # output map=32*32, filter=16
        self.allLayers.append(        
            ConvLayer(rng=rng, 
                      input=input_layer, 
                      filter_shape=(16, 3, 3, 3), 
                      image_shape=(batch_size, 3, 32, 32),
                      activation=T.nnet.nnet.relu))

        self.allLayers.append(        
            ConvLayer(rng=rng, 
                      input=self.allLayers[-1].output, 
                      filter_shape=(16, 16, 3, 3), 
                      image_shape=(batch_size, 16, 32, 32),
                      activation=T.nnet.nnet.relu))

        self.allLayers.append(        
            ConvLayer(rng=rng, 
                      input=self.allLayers[-1].output, 
                      filter_shape=(16, 16, 3, 3), 
                      image_shape=(batch_size, 16, 32, 32),
                      activation=None))

        self.allLayers.append(        
            ActivationLayer(input=self.allLayers[-1].output+self.allLayers[-3].output, 
                            activation=T.nnet.nnet.relu))
        
        for i in xrange(n-1):              
            self.allLayers.append(        
                ConvLayer(rng=rng, 
                          input=self.allLayers[-1].output, 
                          filter_shape=(16, 16, 3, 3), 
                          image_shape=(batch_size, 16, 32, 32),
                          activation=T.nnet.nnet.relu)) 
    
            self.allLayers.append(        
                ConvLayer(rng=rng, 
                          input=self.allLayers[-1].output, 
                          filter_shape=(16, 16, 3, 3), 
                          image_shape=(batch_size, 16, 32, 32),
                          activation=None))
    
            self.allLayers.append(        
                ActivationLayer(input=self.allLayers[-1].output+self.allLayers[-3].output, 
                                activation=T.nnet.nnet.relu))
                      
        self.allLayers.append(        
            PoolingLayer(input=self.allLayers[-1].output, 
                        ds=(2,2),
                        mode='max'))
 
        self.allLayers.append(        
            DropoutLayer(is_train=training_enabled, 
                         input=self.allLayers[-1].output, 
                         p=1-drop_rate))
                                  
        # output map=16*16, filter=32
        self.allLayers.append(
            ConvLayer(rng=rng, 
                      input=self.allLayers[-1].output, 
                      filter_shape=(32, 16, 3, 3), 
                      image_shape=(batch_size, 16, 16, 16),
                      activation=T.nnet.nnet.relu))                    

        self.allLayers.append(        
            ConvLayer(rng=rng, 
                      input=self.allLayers[-1].output, 
                      filter_shape=(32, 32, 3, 3), 
                      image_shape=(batch_size, 32, 16, 16),
                      activation=None))

        self.allLayers.append(        
            ActivationLayer(input=self.allLayers[-1].output
                            +zero_padding(self.allLayers[-3].output,(batch_size, 16, 16, 16),32),
                            activation=T.nnet.nnet.relu))

        for i in xrange(n-1):                            
            self.allLayers.append(        
                ConvLayer(rng=rng, 
                          input=self.allLayers[-1].output, 
                          filter_shape=(32, 32, 3, 3), 
                          image_shape=(batch_size, 32, 16, 16),
                          activation=T.nnet.nnet.relu)) 
    
            self.allLayers.append(        
                ConvLayer(rng=rng, 
                          input=self.allLayers[-1].output, 
                          filter_shape=(32, 32, 3, 3), 
                          image_shape=(batch_size, 32, 16, 16),
                          activation=None))
    
            self.allLayers.append(        
                ActivationLayer(input=self.allLayers[-1].output+self.allLayers[-3].output, 
                                activation=T.nnet.nnet.relu))

                      
        self.allLayers.append(        
            PoolingLayer(input=self.allLayers[-1].output, 
                        ds=(2,2),
                        mode='max'))

        self.allLayers.append(        
            DropoutLayer(is_train=training_enabled, 
                         input=self.allLayers[-1].output, 
                         p=1-drop_rate))
                                 

        # output map=8*8, filter=64
        self.allLayers.append(        
            ConvLayer(rng=rng, 
                      input=self.allLayers[-1].output, 
                      filter_shape=(64, 32, 3, 3), 
                      image_shape=(batch_size, 32, 8, 8),
                      activation=T.nnet.nnet.relu))

        self.allLayers.append(        
            ConvLayer(rng=rng, 
                      input=self.allLayers[-1].output, 
                      filter_shape=(64, 64, 3, 3), 
                      image_shape=(batch_size, 64, 8, 8),
                      activation=None))

        self.allLayers.append(        
            ActivationLayer(input=self.allLayers[-1].output
                            +zero_padding(self.allLayers[-3].output,(batch_size, 32, 8, 8),64),
                            activation=T.nnet.nnet.relu))
        
        for i in xrange(n-1):  
            self.allLayers.append(        
                ConvLayer(rng=rng, 
                          input=self.allLayers[-1].output, 
                          filter_shape=(64, 64, 3, 3), 
                          image_shape=(batch_size, 64, 8, 8),
                          activation=T.nnet.nnet.relu))         
    
            self.allLayers.append(        
                ConvLayer(rng=rng, 
                          input=self.allLayers[-1].output, 
                          filter_shape=(64, 64, 3, 3), 
                          image_shape=(batch_size, 64, 8, 8),
                          activation=None))
    
            self.allLayers.append(        
                ActivationLayer(input=self.allLayers[-1].output+self.allLayers[-3].output, 
                                activation=T.nnet.nnet.relu))
                      
        self.allLayers.append(        
            PoolingLayer(input=self.allLayers[-1].output, 
                         ds=(8,8),
                         mode='average_exc_pad'))

        self.allLayers.append(        
            DropoutLayer(is_train=training_enabled, 
                         input=self.allLayers[-1].output, 
                         p=1-drop_rate))
                      
        self.allLayers.append(        
            LogisticRegression(input=self.allLayers[-1].output.flatten(2), n_in=64, n_out=10))
        
        self.output = self.allLayers[-1]
        self.params = sum([each.params for each in self.allLayers], [])

class ConvHighway1(object):
    def __init__(self, rng, x, b_T, drop_rate, training_enabled, batch_size):

        # Reshape matrix of rasterized images of shape (batch_size, 32 * 32) to a 4D tensor
        input_layer = x.reshape((batch_size, 3, 32, 32))

        self.allLayers = []
        self.allLayers.append(
            DropoutLayer(is_train=training_enabled, 
                         input=input_layer, 
                         p=1-drop_rate))

        self.allLayers.append(        
            ConvLayer(rng=rng, 
                      input=self.allLayers[-1].output, 
                      filter_shape=(16, 3, 3, 3), 
                      image_shape=(batch_size, 3, 32, 32),
                      activation=T.nnet.nnet.relu))

        self.allLayers.append(        
            ConvHighwayLayer(rng=rng,
                             input=self.allLayers[-1].output, 
                             filter_shape=(16, 16, 3, 3), 
                             image_shape=(batch_size, 16, 32, 32),
                             b_T=b_T, 
                             activation=None))

        self.allLayers.append(        
            ConvHighwayLayer(rng=rng,
                             input=self.allLayers[-1].output, 
                             filter_shape=(16, 16, 3, 3), 
                             image_shape=(batch_size, 16, 32, 32),
                             b_T=b_T, 
                             activation=None))

        self.allLayers.append(        
            PoolingLayer(input=self.allLayers[-1].output, 
                         ds=(2,2),
                         mode='max'))

        self.allLayers.append(        
            DropoutLayer(is_train=training_enabled,
                         input=self.allLayers[-1].output, 
                         p=1-drop_rate))

        self.allLayers.append(        
            ConvLayer(rng=rng, 
                      input=self.allLayers[-1].output, 
                      filter_shape=(32, 16, 3, 3), 
                      image_shape=(batch_size, 16, 16, 16),
                      activation=T.nnet.nnet.relu))

        self.allLayers.append(        
            ConvHighwayLayer(rng=rng,
                             input=self.allLayers[-1].output, 
                             filter_shape=(32, 32, 3, 3), 
                             image_shape=(batch_size, 32, 16, 16),
                             b_T=b_T, 
                             activation=None))

        self.allLayers.append(        
            ConvHighwayLayer(rng=rng,
                             input=self.allLayers[-1].output, 
                             filter_shape=(32, 32, 3, 3), 
                             image_shape=(batch_size, 32, 16, 16),
                             b_T=b_T, 
                             activation=None))

        self.allLayers.append(        
            PoolingLayer(input=self.allLayers[-1].output, 
                                  ds=(2,2),
                                  mode='max'))

        self.allLayers.append(        
            DropoutLayer(is_train=training_enabled, 
                                  input=self.allLayers[-1].output, 
                                  p=1-drop_rate))

        self.allLayers.append(        
            ConvLayer(rng=rng, 
                      input=self.allLayers[-1].output, 
                      filter_shape=(64, 32, 3, 3), 
                      image_shape=(batch_size, 32, 8, 8),
                      activation=T.nnet.nnet.relu))

        self.allLayers.append(        
            ConvHighwayLayer(rng=rng,
                             input=self.allLayers[-1].output, 
                             filter_shape=(64, 64, 3, 3), 
                             image_shape=(batch_size, 64, 8, 8),
                             b_T=b_T, 
                             activation=None))

        self.allLayers.append(        
            ConvHighwayLayer(rng=rng,
                             input=self.allLayers[-1].output, 
                             filter_shape=(64, 64, 3, 3), 
                             image_shape=(batch_size, 64, 8, 8),
                             b_T=b_T, 
                             activation=None))

        self.allLayers.append(        
            ConvLayer(rng=rng, 
                      input=self.allLayers[-1].output, 
                      filter_shape=(64, 64, 1, 1), 
                      image_shape=(batch_size, 64, 8, 8),
                      activation=T.nnet.nnet.relu))

        self.allLayers.append(        
            PoolingLayer(input=self.allLayers[-1].output, 
                         ds=(8,8),
                         mode='average_exc_pad'))

        self.allLayers.append(        
            DropoutLayer(is_train=training_enabled, 
                         input=self.allLayers[-1].output, 
                         p=1-drop_rate))

        self.allLayers.append(        
            LogisticRegression(input=self.allLayers[-1].output.flatten(2), n_in=64, n_out=10))
        
        self.output = self.allLayers[-1]
        self.params = sum([each.params for each in self.allLayers], [])

class ConvHighway2(object):
    def __init__(self, rng, x, b_T, drop_rate, training_enabled, batch_size):

        # Reshape matrix of rasterized images of shape (batch_size, 32 * 32) to a 4D tensor
        input_layer = x.reshape((batch_size, 3, 32, 32))

        self.allLayers = []
        self.allLayers.append(
            DropoutLayer(is_train=training_enabled, 
                         input=input_layer, 
                         p=1-drop_rate))

        self.allLayers.append(        
            ConvLayer(rng=rng, 
                      input=self.allLayers[-1].output, 
                      filter_shape=(32, 3, 3, 3), 
                      image_shape=(batch_size, 3, 32, 32),
                      activation=T.nnet.nnet.relu))

        for i in range(4):
            self.allLayers.append(        
                ConvHighwayLayer(rng=rng,
                                 input=self.allLayers[-1].output, 
                                 filter_shape=(32, 32, 3, 3), 
                                 image_shape=(batch_size, 32, 32, 32),
                                 b_T=b_T, 
                                 activation=None))

        self.allLayers.append(        
            PoolingLayer(input=self.allLayers[-1].output, 
                         ds=(2,2),
                         mode='max'))

        self.allLayers.append(        
            DropoutLayer(is_train=training_enabled,
                         input=self.allLayers[-1].output, 
                         p=1-drop_rate))

        self.allLayers.append(        
            ConvLayer(rng=rng, 
                      input=self.allLayers[-1].output, 
                      filter_shape=(80, 32, 3, 3), 
                      image_shape=(batch_size, 32, 16, 16),
                      activation=T.nnet.nnet.relu))
        
        for i in range(5):
            self.allLayers.append(        
                ConvHighwayLayer(rng=rng,
                                 input=self.allLayers[-1].output, 
                                 filter_shape=(80, 80, 3, 3), 
                                 image_shape=(batch_size, 80, 16, 16),
                                 b_T=b_T, 
                                 activation=None))

        self.allLayers.append(        
            PoolingLayer(input=self.allLayers[-1].output, 
                                  ds=(2,2),
                                  mode='max'))

        self.allLayers.append(        
            DropoutLayer(is_train=training_enabled, 
                                  input=self.allLayers[-1].output, 
                                  p=1-drop_rate))

        self.allLayers.append(        
            ConvLayer(rng=rng, 
                      input=self.allLayers[-1].output, 
                      filter_shape=(128, 80, 3, 3), 
                      image_shape=(batch_size, 80, 8, 8),
                      activation=T.nnet.nnet.relu))
        
        for i in range(5):
            self.allLayers.append(        
                ConvHighwayLayer(rng=rng,
                                 input=self.allLayers[-1].output, 
                                 filter_shape=(128, 128, 3, 3), 
                                 image_shape=(batch_size, 128, 8, 8),
                                 b_T=b_T, 
                                 activation=None))

        self.allLayers.append(        
            ConvLayer(rng=rng, 
                      input=self.allLayers[-1].output, 
                      filter_shape=(100, 128, 1, 1), 
                      image_shape=(batch_size, 128, 8, 8),
                      activation=T.nnet.nnet.relu))
                      
        self.allLayers.append(        
            PoolingLayer(input=self.allLayers[-1].output, 
                         ds=(8,8),
                         mode='average_exc_pad'))

        self.allLayers.append(        
            DropoutLayer(is_train=training_enabled, 
                         input=self.allLayers[-1].output, 
                         p=1-drop_rate))

        self.allLayers.append(        
            LogisticRegression(input=self.allLayers[-1].output.flatten(2), n_in=100, n_out=10))
        
        self.output = self.allLayers[-1]
        self.params = sum([each.params for each in self.allLayers], [])

def test_ConvHighway(datasets, model, learning_rate=0.01, lr_decay=0.1, momentum=0.9, step_values = [100000, 150000, 175000], 
                     n_epochs=100, b_T=-2, drop_rate=0.25, batch_size=20, verbose=True):
 
    rng = numpy.random.RandomState(1234)

    train_set_x, train_set_y = datasets[0]
    test_set_x, test_set_y = datasets[1]

    # compute number of minibatches for training and testing
    n_train_batches = train_set_x.get_value(borrow=True).shape[0]
    n_test_batches = test_set_x.get_value(borrow=True).shape[0]
    
    n_in = train_set_x.get_value(borrow=True).shape[1]
    
    n_train_batches //= batch_size
    n_test_batches //= batch_size

    # allocate symbolic variables for the data
    index = T.lscalar()  # index to a [mini]batch
    itr = T.fscalar()  # index to an iteration

    # start-snippet-1
    x = T.matrix('x')   # the data is presented as rasterized images
    y = T.ivector('y')  # the labels are presented as 1D vector of
                        # [int] labels

    training_enabled = T.iscalar('training_enabled') # pseudo boolean for switching between training and prediction
        
    ######################
    # BUILD ACTUAL MODEL #
    ######################
    print('... building the model')
    
    if model==1:
        highway_net = ConvHighway1(rng=rng, 
                                   x=x, 
                                   b_T=b_T, 
                                   drop_rate=drop_rate, 
                                   training_enabled=training_enabled,
                                   batch_size=batch_size)
    elif model==2:
        highway_net = ConvHighway2(rng=rng, 
                                   x=x, 
                                   b_T=b_T, 
                                   drop_rate=drop_rate, 
                                   training_enabled=training_enabled,
                                   batch_size=batch_size)
    elif model==11:
        highway_net = ResNet(rng=rng, 
                              x=x,
                              n=3,
                              drop_rate=drop_rate, 
                              training_enabled=training_enabled,
                              batch_size=batch_size)
    elif model==12:
        highway_net = ResNet(rng=rng, 
                              x=x,
                              n=5,
                              drop_rate=drop_rate, 
                              training_enabled=training_enabled,
                              batch_size=batch_size)
    elif model==13:
        highway_net = ResNet(rng=rng, 
                              x=x,
                              n=7,
                              drop_rate=drop_rate, 
                              training_enabled=training_enabled,
                              batch_size=batch_size)
    elif model==14:
        highway_net = ResNet(rng=rng, 
                              x=x,
                              n=9,
                              drop_rate=drop_rate, 
                              training_enabled=training_enabled,
                              batch_size=batch_size)
    else:
        raise NotImplementedError()

    output_layer = highway_net.output

    # the cost we minimize during training 
    cost = output_layer.negative_log_likelihood(y)
    
    # create a function to compute the mistakes that are made by the model
    test_model = theano.function(
        inputs=[index],
        outputs=output_layer.errors(y),
        givens={
            x: test_set_x[index * batch_size:(index + 1) * batch_size],
            y: test_set_y[index * batch_size:(index + 1) * batch_size],
            training_enabled: numpy.cast['int32'](0)
        },
        on_unused_input='ignore'
    )

    # SGD momentum with multi-step decay
    updates = MomentumWithMultiStepDecay(cost, highway_net.params, itr, lr_base=learning_rate, 
                                         lr_decay=lr_decay, momentum=momentum, step_values=step_values)     

    # augmentation
    train_set_new_x = theano.shared(train_set_x.get_value(borrow=False), borrow=True)
    def augment_data():
        flipped = flip_image(rng, train_set_x.get_value(borrow=True))        
        rotated = rotate_image(rng, flipped)        
        translated = translate_image(rng, rotated)
        train_set_new_x.set_value(translated,borrow=True)
        
    train_model = theano.function(
        inputs=[index,itr],
        outputs=cost,
        updates=updates,
        givens={
            x: train_set_new_x[index * batch_size: (index + 1) * batch_size],
            y: train_set_y[index * batch_size: (index + 1) * batch_size],
            training_enabled: numpy.cast['int32'](1)
        },
        on_unused_input='ignore'        
    ) 
        
    # train the model
    train_nn_NoValidation(train_model, test_model, n_train_batches, n_test_batches, n_epochs, verbose, augment_data)
