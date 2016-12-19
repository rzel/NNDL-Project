"""
This code contains implementation of some basic components in neural network.
This code is based on
[1] http://deeplearning.net/tutorial/logreg.html
[2] http://deeplearning.net/tutorial/mlp.html
[3] http://deeplearning.net/tutorial/rnnslu.html
"""

from __future__ import print_function

import os
import timeit
import inspect
import sys
import numpy
import scipy.io
import tarfile
from collections import OrderedDict

import theano
import theano.tensor as T
from theano.tensor.nnet import conv2d
from theano.tensor.signal import pool

import gzip
import pickle
from theano.tensor.signal import downsample

class LogisticRegression(object):
    """Multi-class Logistic Regression Class
    The logistic regression is fully described by a weight matrix :math:`W`
    and bias vector :math:`b`. Classification is done by projecting data
    points onto a set of hyperplanes, the distance to which is used to
    determine a class membership probability.
    """

    def __init__(self, input, n_in, n_out):
        """ Initialize the parameters of the logistic regression
        :type input: theano.tensor.TensorType
        :param input: symbolic variable that describes the input of the
                      architecture (one minibatch)
        :type n_in: int
        :param n_in: number of input units, the dimension of the space in
                     which the datapoints lie
        :type n_out: int
        :param n_out: number of output units, the dimension of the space in
                      which the labels lie
        """
        # initialize with 0 the weights W as a matrix of shape (n_in, n_out)
        self.W = theano.shared(
            value=numpy.ones(
                (n_in, n_out),
                dtype=theano.config.floatX
            ),
            name='W',
            borrow=True
        )
        # initialize the biases b as a vector of n_out 0s
        self.b = theano.shared(
            value=0.1*numpy.ones(
                (n_out,),
                dtype=theano.config.floatX
            ),
            name='b',
            borrow=True
        )
        
        # symbolic expression for computing the matrix of class-membership
        # probabilities
        # Where:
        # W is a matrix where column-k represent the separation hyperplane for
        # class-k
        # x is a matrix where row-j  represents input training sample-j
        # b is a vector where element-k represent the free parameter of
        # hyperplane-k
        # self.p_y_given_x = T.nnet.nnet.relu(T.dot(input, self.W) + self.b)
        self.p_y_given_x = T.nnet.softmax(T.dot(input, self.W) + self.b)

        # symbolic description of how to compute prediction as class whose
        # probability is maximal
        self.y_pred = T.argmax(self.p_y_given_x, axis=1)

        # parameters of the model
        self.params = [self.W, self.b]

        # keep track of model input
        self.input = input

    def negative_log_likelihood(self, y):
        """Return the mean of the negative log-likelihood of the prediction
        of this model under a given target distribution.
        .. math::
            \frac{1}{|\mathcal{D}|} \mathcal{L} (\theta=\{W,b\}, \mathcal{D}) =
            \frac{1}{|\mathcal{D}|} \sum_{i=0}^{|\mathcal{D}|}
                \log(P(Y=y^{(i)}|x^{(i)}, W,b)) \\
            \ell (\theta=\{W,b\}, \mathcal{D})
        :type y: theano.tensor.TensorType
        :param y: corresponds to a vector that gives for each example the
                  correct label
        Note: we use the mean instead of the sum so that
              the learning rate is less dependent on the batch size
        """
        # y.shape[0] is (symbolically) the number of rows in y, i.e.,
        # number of examples (call it n) in the minibatch
        # T.arange(y.shape[0]) is a symbolic vector which will contain
        # [0,1,2,... n-1] T.log(self.p_y_given_x) is a matrix of
        # Log-Probabilities (call it LP) with one row per example and
        # one column per class LP[T.arange(y.shape[0]),y] is a vector
        # v containing [LP[0,y[0]], LP[1,y[1]], LP[2,y[2]], ...,
        # LP[n-1,y[n-1]]] and T.mean(LP[T.arange(y.shape[0]),y]) is
        # the mean (across minibatch examples) of the elements in v,
        # i.e., the mean log-likelihood across the minibatch.
        return -T.mean(T.log(self.p_y_given_x)[T.arange(y.shape[0]), y])

    def errors(self, y):
        """Return a float representing the number of errors in the minibatch
        over the total number of examples of the minibatch ; zero one
        loss over the size of the minibatch
        :type y: theano.tensor.TensorType
        :param y: corresponds to a vector that gives for each example the
                  correct label
        """

        # check if y has same dimension of y_pred
        if y.ndim != self.y_pred.ndim:
            raise TypeError(
                'y should have the same shape as self.y_pred',
                ('y', y.type, 'y_pred', self.y_pred.type)
            )
        # check if y is of the correct datatype
        if y.dtype.startswith('int'):
            # the T.neq operator returns a vector of 0s and 1s, where 1
            # represents a mistake in prediction
            return T.mean(T.neq(self.y_pred, y))
        else:
            raise NotImplementedError()

class HiddenLayer(object):
    def __init__(self, rng, input, n_in, n_out, W=None, b=None,
                 activation=T.tanh):
        """
        Typical hidden layer of a MLP: units are fully-connected and have
        sigmoidal activation function. Weight matrix W is of shape (n_in,n_out)
        and the bias vector b is of shape (n_out,).
        NOTE : The nonlinearity used here is tanh
        Hidden unit activation is given by: tanh(dot(input,W) + b)
        :type rng: numpy.random.RandomState
        :param rng: a random number generator used to initialize weights
        :type input: theano.tensor.dmatrix
        :param input: a symbolic tensor of shape (n_examples, n_in)
        :type n_in: int
        :param n_in: dimensionality of input
        :type n_out: int
        :param n_out: number of hidden units
        :type activation: theano.Op or function
        :param activation: Non linearity to be applied in the hidden
                           layer
        """
        self.input = input

        # `W` is initialized with `W_values` which is uniformely sampled
        # from sqrt(-6./(n_in+n_hidden)) and sqrt(6./(n_in+n_hidden))
        # for tanh activation function
        # the output of uniform if converted using asarray to dtype
        # theano.config.floatX so that the code is runable on GPU
        # Note : optimal initialization of weights is dependent on the
        #        activation function used (among other things).
        #        For example, results presented in [Xavier10] suggest that you
        #        should use 4 times larger initial weights for sigmoid
        #        compared to tanh
        #        We have no info for other function, so we use the same as
        #        tanh.
        if W is None:
            W_values = numpy.asarray(
                rng.uniform(
                    low=-numpy.sqrt(6. / (n_in + n_out)),
                    high=numpy.sqrt(6. / (n_in + n_out)),
                    size=(n_in, n_out)
                ),
                dtype=theano.config.floatX
            )
            if activation == theano.tensor.nnet.sigmoid:
                W_values *= 4

            W = theano.shared(value=W_values, name='W', borrow=True)

        if b is None:
            b_values = numpy.zeros((n_out,), dtype=theano.config.floatX)
            b = theano.shared(value=b_values, name='b', borrow=True)

        self.W = W
        self.b = b

        lin_output = T.dot(input, self.W) + self.b
        self.output = (
            lin_output if activation is None
            else activation(lin_output)
        )
        # parameters of the model
        self.params = [self.W, self.b]

class ConvLayer(object):
    """ pure convolutional layer """

    def __init__(self, rng, input, filter_shape, image_shape,
                 activation=T.tanh):
        """
        Allocate a pure convolutional layer with shared variable internal parameters.
        :type rng: numpy.random.RandomState
        :param rng: a random number generator used to initialize weights
        :type input: theano.tensor.dtensor4
        :param input: symbolic image tensor, of shape image_shape
        :type filter_shape: tuple or list of length 4
        :param filter_shape: (number of filters, num input feature maps,
                              filter height, filter width)
        :type image_shape: tuple or list of length 4
        :param image_shape: (batch size, num input feature maps,
                             image height, image width)
        """
        assert image_shape[1] == filter_shape[1]

        # there are "num input feature maps * filter height * filter width"
        # inputs to each hidden unit
        fan_in = numpy.prod(filter_shape[1:])
        # each unit in the lower layer receives a gradient from:
        # "num output feature maps * filter height * filter width" 
        fan_out = (filter_shape[0] * numpy.prod(filter_shape[2:]))
        # initialize weights with random weights
        W_bound = numpy.sqrt(6. / (fan_in + fan_out))
        self.W = theano.shared(
            numpy.asarray(
                rng.uniform(low=-W_bound, high=W_bound, size=filter_shape),
                dtype=theano.config.floatX
            ),
            borrow=True
        )

        # the bias is a 1D tensor -- one bias per output feature map
        if activation==T.nnet.relu:
            b_values = numpy.ones((filter_shape[0],), dtype=theano.config.floatX)*0.1            
        else:
            b_values = numpy.zeros((filter_shape[0],), dtype=theano.config.floatX)
        self.b = theano.shared(value=b_values, borrow=True)

        # convolve input feature maps with filters
        conv_out = conv2d(
            input=input,
            filters=self.W,
            filter_shape=filter_shape,
            input_shape=image_shape,
            border_mode='half'
        )

        # add the bias term. Since the bias is a vector (1D array), we first
        # reshape it to a tensor of shape (1, n_filters, 1, 1). Each bias will
        # thus be broadcasted across mini-batches and feature map
        # width & height
        if activation is None:
            self.output = conv_out + self.b.dimshuffle('x', 0, 'x', 'x')
        else:
            self.output = activation(conv_out + self.b.dimshuffle('x', 0, 'x', 'x'))

        # store parameters of this layer
        self.params = [self.W, self.b]

        # keep track of model input
        self.input = input    

class ConvHighwayLayer(object):
    """ Convolutional highway layer """

    def __init__(self, rng, input, filter_shape, image_shape, b_T=None, activation=T.tanh):
        """
        Allocate a convolutional highway layer with shared variable internal parameters.
        :type rng: numpy.random.RandomState
        :param rng: a random number generator used to initialize weights
        :type input: theano.tensor.dtensor4
        :param input: symbolic image tensor, of shape image_shape
        :type filter_shape: tuple or list of length 4
        :param filter_shape: (number of filters, num input feature maps,
                              filter height, filter width)
        :type image_shape: tuple or list of length 4
        :param image_shape: (batch size, num input feature maps,
                             image height, image width)
        """
        assert image_shape[1] == filter_shape[1]

        # there are "num input feature maps * filter height * filter width"
        # inputs to each hidden unit
        fan_in = numpy.prod(filter_shape[1:])
        # each unit in the lower layer receives a gradient from:
        # "num output feature maps * filter height * filter width" 
        fan_out = (filter_shape[0] * numpy.prod(filter_shape[2:]))
        # initialize weights with random weights
        W_bound = numpy.sqrt(6. / (fan_in + fan_out))
 
        self.W_H = theano.shared(
            numpy.asarray(
                rng.uniform(low=-W_bound, high=W_bound, size=filter_shape),
                dtype=theano.config.floatX
            ),
            borrow=True
        )
        
        self.W_T = theano.shared(
            numpy.asarray(
                rng.uniform(low=-W_bound, high=W_bound, size=filter_shape),
                dtype=theano.config.floatX
            ),
            borrow=True
        )
        
        # the bias is a 1D tensor -- one bias per output feature map
        if activation==T.nnet.relu:
            b_H_values = numpy.ones((filter_shape[0],), dtype=theano.config.floatX)*0.1            
        else:
            b_H_values = numpy.zeros((filter_shape[0],), dtype=theano.config.floatX)

        self.b_H = theano.shared(value=b_H_values, borrow=True)

        if b_T is None:
            b_T_values = numpy.zeros((filter_shape[0],), dtype=theano.config.floatX)
        else:
            b_T_values = b_T * numpy.ones((filter_shape[0],), dtype=theano.config.floatX)

        self.b_T = theano.shared(value=b_T_values, borrow=True)

        # convolve input feature maps with filters
        conv_out_H = conv2d(
            input=input,
            filters=self.W_H,
            filter_shape=filter_shape,
            input_shape=image_shape,
            border_mode='half'
        )
        conv_out_T = conv2d(
            input=input,
            filters=self.W_T,
            filter_shape=filter_shape,
            input_shape=image_shape,
            border_mode='half'
        )

        # add the bias term. Since the bias is a vector (1D array), we first
        # reshape it to a tensor of shape (1, n_filters, 1, 1). Each bias will
        # thus be broadcasted across mini-batches and feature map
        # width & height
        H_part = conv_out_H + self.b_H.dimshuffle('x', 0, 'x', 'x')
        T_part = conv_out_T + self.b_T.dimshuffle('x', 0, 'x', 'x')
        
        if activation is None:
            self.output = H_part*T.nnet.nnet.sigmoid(T_part) + input-input*T.nnet.nnet.sigmoid(T_part)
        else:
            self.output = activation(H_part)*T.nnet.nnet.sigmoid(T_part) + input-input*T.nnet.nnet.sigmoid(T_part)

        # store parameters of this layer
        self.params = [self.W_H, self.b_H, self.W_T, self.b_T]

        # keep track of model input
        self.input = input  
        
class HighwayLayer(object):
    def __init__(self, rng, input, n_in, n_out, W_H=None, b_H=None, W_T =None, b_T = None,
                 activation_H=T.tanh, activation_T = T.nnet.nnet.sigmoid):
        self.input = input

        if W_H is None:
            W_H_values = numpy.asarray(rng.uniform(
                    low=-numpy.sqrt(6. / (n_in + n_out)),
                    high=numpy.sqrt(6. / (n_in + n_out)),
                    size=(n_in, n_out)),dtype=theano.config.floatX)
            if activation_H == theano.tensor.nnet.sigmoid:
                W_H_values *= 4
            W_H = theano.shared(value=W_H_values, name='W_H', borrow=True)

        if b_H is None:
            b_H_values = numpy.zeros((n_out,), dtype=theano.config.floatX)
            b_H = theano.shared(value=b_H_values, name='b_H', borrow=True)
    
        if W_T is None:
            W_T_values = numpy.asarray(rng.uniform(
                    low=-numpy.sqrt(6. / (n_in + n_out)),
                    high=numpy.sqrt(6. / (n_in + n_out)),
                    size=(n_in, n_out)),dtype=theano.config.floatX)
            if activation_T == theano.tensor.nnet.sigmoid:
                W_T_values *= 4
            W_T = theano.shared(value=W_T_values, name='W_T', borrow=True)

        if b_T is None:
            b_T_values = numpy.asarray(rng.uniform(low=-10,high=-1,size=(n_out,)),dtype=theano.config.floatX)
            b_T = theano.shared(value=b_T_values, name='b_T', borrow=True)
        else:
            b_T_values = numpy.asarray(b_T*numpy.ones(shape=(n_out,)),dtype=theano.config.floatX)
            b_T = theano.shared(value=b_T_values,name='b_T',borrow=True)
        
        self.W_H = W_H
        self.b_H = b_H
        self.W_T = W_T
        self.b_T = b_T

        H_part = T.dot(input, self.W_H) + self.b_H
        T_part = T.dot(input, self.W_T) + self.b_T
    
        one = numpy.ones(n_in,dtype=theano.config.floatX)
        self.output = activation_H(H_part)*activation_T(T_part) + input*(one-activation_T(T_part))
        # parameters of the model
        self.params = [self.W_H, self.W_T, self.b_H, self.b_T]

class HighwayNetwork(object):
    def __init__(self, rng, input, n_in, n_hidden, n_out, n_hiddenLayers, n_highwayLayers, activation_hidden, activation_highway, b_T):#, training_enabled):

        if hasattr(n_hidden, '__iter__'):
            assert(len(n_hidden) == n_hiddenLayers)
        else:
            n_hidden = (n_hidden,)*n_hiddenLayers

        # fully connected layers
        self.allLayers = []
        for i in xrange(n_hiddenLayers):
            h_input = input if i == 0 else self.allLayers[i-1].output
            h_in = n_in if i == 0 else n_hidden[i-1]
            self.allLayers.append(
                HiddenLayer(
                    rng = rng,
                    input = h_input,
                    n_in = h_in,
                    n_out = n_hidden[i],
                    activation = activation_hidden
                ))

        # highway layers
        h_in = n_in if n_hiddenLayers == 0 else n_hidden[-1]
        for i in range(n_hiddenLayers,n_hiddenLayers+n_highwayLayers):
            h_input = input if n_hiddenLayers==0 else self.allLayers[i-1].output
            self.allLayers.append(
                HighwayLayer(
                    rng = rng, 
                    input = h_input,
                    n_in = h_in,
                    n_out = h_in, 
                    b_T = b_T,
                    activation_H = activation_highway,
                ))

        # logistic regression layer
        self.logRegressionLayer = LogisticRegression(
            input = self.allLayers[-1].output,
            n_in = h_in,
            n_out = n_out
        )

        self.negative_log_likelihood = (
            self.logRegressionLayer.negative_log_likelihood
        )

        self.errors = self.logRegressionLayer.errors

        self.params = sum([x.params for x in self.allLayers], []) + self.logRegressionLayer.params
                                       
        self.input = input

class myMLP(object):
    """Multi-Layefr Perceptron Class
    A multilayer perceptron is a feedforward artificial neural network model
    that has one layer or more of hidden units and nonlinear activations.
    Intermediate layers usually have as activation function tanh or the
    sigmoid function (defined here by a ``HiddenLayer`` class)  while the
    top layer is a softmax layer (defined here by a ``LogisticRegression``
    class).
    """

    def __init__(self, rng, input, n_in, n_hidden, n_out, n_hiddenLayers, activation):
        """Initialize the parameters for the multilayer perceptron
        :type rng: numpy.random.RandomState
        :param rng: a random number generator used to initialize weights
        :type input: theano.tensor.TensorType
        :param input: symbolic variable that describes the input of the
        architecture (one minibatch)
        :type n_in: int
        :param n_in: number of input units, the dimension of the space in
        which the datapoints lie
        :type n_hidden: int or list of ints
        :param n_hidden: number of hidden units. If a list, it specifies the
        number of units in each hidden layers, and its length should equal to
        n_hiddenLayers.
        :type n_out: int
        :param n_out: number of output units, the dimension of the space in
        which the labels lie
        :type n_hiddenLayers: int
        :param n_hiddenLayers: number of hidden layers
        """
        # If n_hidden is a list (or tuple), check its length is equal to the
        # number of hidden layers. If n_hidden is a scalar, we set up every
        # hidden layers with same number of units.
        if hasattr(n_hidden, '__iter__'):
            assert(len(n_hidden) == n_hiddenLayers)
        else:
            n_hidden = (n_hidden,)*n_hiddenLayers

        # Since we are dealing with a one hidden layer MLP, this will translate
        # into a HiddenLayer with a tanh activation function connected to the
        # LogisticRegression layer; the activation function can be replaced by
        # sigmoid or any other nonlinear function.
        self.hiddenLayers = []
        for i in xrange(n_hiddenLayers):
            h_input = input if i == 0 else self.hiddenLayers[i-1].output
            h_in = n_in if i == 0 else n_hidden[i-1]
            self.hiddenLayers.append(
                HiddenLayer(
                    rng=rng,
                    input=h_input,
                    n_in=h_in,
                    n_out=n_hidden[i],
                    activation=activation
            ))

        # The logistic regression layer gets as input the hidden units
        # of the hidden layer
        self.logRegressionLayer = LogisticRegression(
            input=self.hiddenLayers[-1].output,
            n_in= n_hidden[n_hiddenLayers-1], #n_hidden[-1],
            n_out=n_out
        )
        # L1 norm ; one regularization option is to enforce L1 norm to
        # be small
        self.L1 = (
            sum([abs(x.W).sum() for x in self.hiddenLayers])
            + abs(self.logRegressionLayer.W).sum()
        )

        # square of L2 norm ; one regularization option is to enforce
        # square of L2 norm to be small
        self.L2_sqr = (
            sum([(x.W ** 2).sum() for x in self.hiddenLayers])
            + (self.logRegressionLayer.W ** 2).sum()
        )

        # negative log likelihood of the MLP is given by the negative
        # log likelihood of the output of the model, computed in the
        # logistic regression layer
        self.negative_log_likelihood = (
            self.logRegressionLayer.negative_log_likelihood
        )
        # same holds for the function computing the number of errors
        self.errors = self.logRegressionLayer.errors

        # the parameters of the model are the parameters of the two layer it is
        # made out of
        self.params = sum([x.params for x in self.hiddenLayers], []) + self.logRegressionLayer.params
        #print(self.params)

        # keep track of model input
        self.input = input

class ActivationLayer(object):
    """ pure activation layer """
    def __init__(self, input, activation=T.tanh):
        self.input = input
        self.output = activation(input)
        self.params = []
                         
def drop(input, p=0.5): 
    """
    :type input: numpy.array
    :param input: layer or weight matrix on which dropout is applied
    
    :type p: float or double between 0. and 1. 
    :param p: p probability of NOT dropping out a unit, therefore (1.-p) is the drop rate.
    
    """            
    rng = numpy.random.RandomState(5112)
    srng = T.shared_randomstreams.RandomStreams(rng.randint(999999))
    mask = srng.binomial(n=1, p=p, size=input.shape, dtype=theano.config.floatX)
    return input * mask


class DropoutLayer:
    def __init__(self, is_train, input, p=0.5):
        """
        Dropout layer class
        
        :type is_train: theano.iscalar   
        :param is_train: indicator pseudo-boolean (int) for switching between training and prediction
        
        :type input: theano.tensor.dmatrix
        :param input: a symbolic tensor
                           
        :type p: float or double
        :param p: probability of NOT dropping out a unit          
        """
        self.input = input
        self.output = T.switch(T.neq(is_train, 0), drop(input, p), p*input)
        self.params = []
        
class PoolingLayer(object):
    def __init__(self, input, ds, mode='max'):
        """
        Pooling layer class
        
        :type input: theano.tensor.dmatrix
        :param input: a symbolic tensor
        :type ds: tuple of length 2
        :param ds: a factor by which to downscale
        :type mode: string
        :param mode: pooling mode {'max', 'sum', 'average_inc_pad', 'average_exc_pad'}
        """
        self.input = input
        self.output = pool.pool_2d(input=input, ds=ds, ignore_border=True, mode=mode)
        self.params = []
    
class DropoutHiddenLayer(object):
    def __init__(self, rng, is_train, input, n_in, n_out, p=0.5, W=None, b=None,
                 activation=T.nnet.nnet.relu):
        """
        Hidden unit activation is given by: activation(dot(input,W) + b)
        :type rng: numpy.random.RandomState
        :param rng: a random number generator used to initialize weights
        
        :type is_train: theano.iscalar   
        :param is_train: indicator pseudo-boolean (int) for switching between training and prediction
        :type input: theano.tensor.dmatrix
        :param input: a symbolic tensor of shape (n_examples, n_in)
        :type n_in: int
        :param n_in: dimensionality of input
        :type n_out: int
        :param n_out: number of hidden units
        :type activation: theano.Op or function
        :param activation: Non linearity to be applied in the hidden
                           layer
                           
        :type p: float or double
        :param p: probability of NOT dropping out a unit   
        """
        self.input = input

        if W is None:
            W_values = numpy.asarray(
                rng.uniform(
                    low=-numpy.sqrt(6. / (n_in + n_out)),
                    high=numpy.sqrt(6. / (n_in + n_out)),
                    size=(n_in, n_out)
                ),
                dtype=theano.config.floatX
            )
            if activation == theano.tensor.nnet.sigmoid:
                W_values *= 4

            W = theano.shared(value=W_values, name='W', borrow=True)

        if b is None:
            b_values = numpy.zeros((n_out,), dtype=theano.config.floatX)
            b = theano.shared(value=b_values, name='b', borrow=True)

        
        self.W = W
        self.b = b

        lin_output = T.dot(input, self.W) + self.b
        
        output = activation(lin_output)
        
        # multiply output and drop -> in an approximation the scaling effects cancel out 
        train_output = drop(output,p)
        
        #is_train is a pseudo boolean theano variable for switching between training and prediction 
        self.output = T.switch(T.neq(is_train, 0), train_output, p*output)
        
        # parameters of the model
        self.params = [self.W, self.b]  

class train_result(object):
    """Result Class
    This is a class to store all the training results
    """
    def __init__(self, RunningTime, XEntropy, TestPerformance, BestValidationScore, N_Epochs, N_Iterations, Patience):
        self.RunningTime = RunningTime
        self.XEntropy = XEntropy
        self.TestPerformance = TestPerformance
        self.BestValidationScore = BestValidationScore
        self.N_Epochs = N_Epochs
        self.N_Iterations = N_Iterations
        self.Patience = Patience
        
        self.BestXEntropy = XEntropy[N_Epochs]
        
def doNothing():
    return 0
    
def train_nn(train_model, validate_model, test_model, 
            n_train_batches, n_valid_batches, n_test_batches, n_epochs, verbose = True):
    """
    Wrapper function for training and test THEANO model
    :type train_model: Theano.function
    :param train_model:
    :type validate_model: Theano.function
    :param validate_model:
    :type test_model: Theano.function
    :param test_model:
    :type n_train_batches: int
    :param n_train_batches: number of training batches
    :type n_valid_batches: int
    :param n_valid_batches: number of validation batches
    :type n_test_batches: int
    :param n_test_batches: number of testing batches
    :type n_epochs: int
    :param n_epochs: maximal number of epochs to run the optimizer
    :type verbose: boolean
    :param verbose: to print out epoch summary or not to
    """

    # early-stopping parameters
    patience = 10000  # look as this many examples regardless
    patience_increase = 10  # wait this much longer when a new best is
                           # found
    improvement_threshold = 0.9995  # a relative improvement of this much is
                                   # considered significant
    validation_frequency = min(n_train_batches, patience // 2)
                                  # go through this many
                                  # minibatche before checking the network
                                  # on the validation set; in this case we
                                  # check every epoch

    best_validation_loss = numpy.inf
    best_iter = 0
    test_score = 0.
    start_time = timeit.default_timer()

    epoch = 0
    done_looping = False

    cross_entropy = {}
    
    while (epoch < n_epochs) and (not done_looping):
        epoch = epoch + 1
        epoch_cross_entropy = 0
        for minibatch_index in range(n_train_batches):

            iter = (epoch - 1) * n_train_batches + minibatch_index

            #if (iter % 100 == 0) and verbose:
            #    print('training @ iter = ', iter)
            cost_ij = train_model(minibatch_index, iter)
            
            epoch_cross_entropy = epoch_cross_entropy + cost_ij
            
            #cross_entropy[(epoch,minibatch_index)] = cost_ij
            
            
            if (iter + 1) % validation_frequency == 0:

                # compute zero-one loss on validation set
                validation_losses = [validate_model(i) for i
                                     in range(n_valid_batches)]
                this_validation_loss = numpy.mean(validation_losses)

                # if we got the best validation score until now
                if this_validation_loss < best_validation_loss:
                    if verbose:
                        print('epoch %i, minibatch %i/%i, validation error %f %%' %
                            (epoch,
                             minibatch_index + 1,
                             n_train_batches,
                             this_validation_loss * 100.))

                    #improve patience if loss improvement is good enough
                    if this_validation_loss < best_validation_loss *  \
                       improvement_threshold:
                        patience = max(patience, iter * patience_increase)

                    # save best validation score and iteration number
                    best_validation_loss = this_validation_loss
                    best_iter = iter

                    # test it on the test set
                    test_losses = [
                        test_model(i)
                        for i in range(n_test_batches)
                    ]
                    test_score = numpy.mean(test_losses)

                    if verbose:
                        print(('     epoch %i, minibatch %i/%i, test error of '
                               'best model %f %%') %
                              (epoch, minibatch_index + 1,
                               n_train_batches,
                               test_score * 100.))

            if patience <= iter:
                done_looping = True
                break
                
        cross_entropy[epoch] = epoch_cross_entropy/n_train_batches
        
    end_time = timeit.default_timer()

    # Retrieve the name of function who invokes train_nn() (caller's name)
    curframe = inspect.currentframe()
    calframe = inspect.getouterframes(curframe, 2)
    
    # Print out summary
    print('Optimization complete.')
    print('Best validation score of %f %% obtained at iteration %i, '
          'with test performance %f %%' %
          (best_validation_loss * 100., best_iter + 1, test_score * 100.))
    print(('The training process for function ' +
           calframe[1][3] +
           ' ran for %.2fm' % ((end_time - start_time) / 60.)), file=sys.stderr)
    
    return train_result(
            RunningTime = (end_time - start_time) / 60.,
            XEntropy = cross_entropy,
            TestPerformance = test_score * 100.,
            BestValidationScore = best_validation_loss * 100.,
            N_Epochs = epoch,
            N_Iterations = best_iter + 1,
            Patience = patience
        ) 

def train_nn_NoValidation(train_model, test_model, n_train_batches, n_test_batches, n_epochs, verbose=True, augment_data=doNothing):
    """
    Wrapper function for training and test THEANO model without validation
    :type train_model: Theano.function
    :param train_model:
    :type test_model: Theano.function
    :param test_model:
    :type n_train_batches: int
    :param n_train_batches: number of training batches
    :type n_test_batches: int
    :param n_test_batches: number of testing batches
    :type n_epochs: int
    :param n_epochs: maximal number of epochs to run the optimizer
    :type verbose: boolean
    :param verbose: to print out epoch summary or not to
    """

    # early-stopping parameters
    patience = 10000  # look as this many examples regardless
    patience_increase = 10  # wait this much longer when a new best is
                           # found
    improvement_threshold = 0.9995  # a relative improvement of this much is
                                   # considered significant
    test_frequency = min(n_train_batches, patience // 2)
                                  # go through this many
                                  # minibatche before checking the network
                                  # on the validation set; in this case we
                                  # check every epoch

    best_test_loss = numpy.inf
    best_iter = 0
    test_score = 0.
    start_time = timeit.default_timer()

    epoch = 0
    done_looping = False

    cross_entropy = {}
    
    while (epoch < n_epochs) and (not done_looping):
        augment_data()
        epoch = epoch + 1
        for minibatch_index in range(n_train_batches):

            iter = (epoch - 1) * n_train_batches + minibatch_index

            cost_ij = train_model(minibatch_index, iter)
            if (iter % 100 == 0) and verbose:
                print('training @ iter = ', iter)
                print(cost_ij)            
            
            if (iter + 1) % test_frequency == 0:

                # compute zero-one loss on test set
                test_losses = [test_model(i) for i
                                     in range(n_test_batches)]
                this_test_loss = numpy.mean(test_losses)

                if verbose:
                    print('epoch %i, minibatch %i/%i, test error %f %%' %
                          (epoch,
                           minibatch_index + 1,
                           n_train_batches,
                           this_test_loss * 100.))
                
                # if we got the best test score until now
                if this_test_loss < best_test_loss:

                    #improve patience if loss improvement is good enough
                    if this_test_loss < best_test_loss *  \
                       improvement_threshold:
                        patience = max(patience, iter * patience_increase)

                    # save best test score and iteration number
                    best_test_loss = this_test_loss
                    best_iter = iter

            if patience <= iter:
                done_looping = True
                break
        
    end_time = timeit.default_timer()

    # Retrieve the name of function who invokes train_nn() (caller's name)
    curframe = inspect.currentframe()
    calframe = inspect.getouterframes(curframe, 2)
    
    # Print out summary
    print('Optimization complete.')
    print('Best test score of %f %% obtained at iteration %i' %
          (best_test_loss * 100., best_iter + 1))
    print(('The training process for function ' +
           calframe[1][3] +
           ' ran for %.2fm' % ((end_time - start_time) / 60.)), file=sys.stderr)
    
    cross_entropy[epoch] = 0.0 # dummmy value
    return train_result(
            RunningTime = (end_time - start_time) / 60.,
            XEntropy = cross_entropy,
            TestPerformance = test_score * 100.,
            BestValidationScore = 0.0,  # dummmy value
            N_Epochs = epoch,
            N_Iterations = best_iter + 1,
            Patience = patience
        ) 

def load_data(dataset):
    ''' Loads the dataset
    :type dataset: string
    :param dataset: the path to the dataset (here MNIST)
    '''
    #############
    # LOAD DATA #
    #############
    # Download the MNIST dataset if it is not present
    data_dir, data_file = os.path.split(dataset)
    if data_dir == "" and not os.path.isfile(dataset):
        # Check if dataset is in the data directory.
        new_path = os.path.join(
            os.path.split(__file__)[0],
            "..",
            "data",
            dataset
        )
        if os.path.isfile(new_path) or data_file == 'mnist.pkl.gz':
            dataset = new_path
    if (not os.path.isfile(dataset)) and data_file == 'mnist.pkl.gz':
        from six.moves import urllib
        origin = (
            'http://www.iro.umontreal.ca/~lisa/deep/data/mnist/mnist.pkl.gz'
        )
        print('Downloading data from %s' % origin)
        urllib.request.urlretrieve(origin, dataset)
    print('... loading data')
    # Load the dataset
    with gzip.open(dataset, 'rb') as f:
        try:
            train_set, valid_set, test_set = pickle.load(f, encoding='latin1')
        except:
            train_set, valid_set, test_set = pickle.load(f)
    # train_set, valid_set, test_set format: tuple(input, target)
    # input is a numpy.ndarray of 2 dimensions (a matrix)
    # where each row corresponds to an example. target is a
    # numpy.ndarray of 1 dimension (vector) that has the same length as
    # the number of rows in the input. It should give the target
    # to the example with the same index in the input.
    def shared_dataset(data_xy, borrow=True):
        """ Function that loads the dataset into shared variables
        The reason we store our dataset in shared variables is to allow
        Theano to copy it into the GPU memory (when code is run on GPU).
        Since copying data into the GPU is slow, copying a minibatch everytime
        is needed (the default behaviour if the data is not in a shared
        variable) would lead to a large decrease in performance.
        """
        data_x, data_y = data_xy
        shared_x = theano.shared(numpy.asarray(data_x,
                                               dtype=theano.config.floatX),
                                 borrow=borrow)
        shared_y = theano.shared(numpy.asarray(data_y,
                                               dtype=theano.config.floatX),
                                 borrow=borrow)
        # When storing data on the GPU it has to be stored as floats
        # therefore we will store the labels as ``floatX`` as well
        # (``shared_y`` does exactly that). But during our computations
        # we need them as ints (we use labels as index, and if they are
        # floats it doesn't make sense) therefore instead of returning
        # ``shared_y`` we will have to cast it to int. This little hack
        # lets ous get around this issue
        return shared_x, T.cast(shared_y, 'int32')
    test_set_x, test_set_y = shared_dataset(test_set)
    valid_set_x, valid_set_y = shared_dataset(valid_set)
    train_set_x, train_set_y = shared_dataset(train_set)
    rval = [(train_set_x, train_set_y), (valid_set_x, valid_set_y),
            (test_set_x, test_set_y)]
    return rval    

def shared_dataset(data_xy, borrow=True):
    """ Function that loads the dataset into shared variables
    The reason we store our dataset in shared variables is to allow
    Theano to copy it into the GPU memory (when code is run on GPU).
    Since copying data into the GPU is slow, copying a minibatch everytime
    is needed (the default behaviour if the data is not in a shared
    variable) would lead to a large decrease in performance.
    """
    data_x, data_y = data_xy
    shared_x = theano.shared(numpy.asarray(data_x,
                                           dtype=theano.config.floatX),
                             borrow=borrow)
    shared_y = theano.shared(numpy.asarray(data_y,
                                           dtype=theano.config.floatX),
                             borrow=borrow)
    # When storing data on the GPU it has to be stored as floats
    # therefore we will store the labels as ``floatX`` as well
    # (``shared_y`` does exactly that). But during our computations
    # we need them as ints (we use labels as index, and if they are
    # floats it doesn't make sense) therefore instead of returning
    # ``shared_y`` we will have to cast it to int. This little hack
    # lets ous get around this issue
    return shared_x, T.cast(shared_y, 'int32')

def load_data_SVHN(ds_rate=None, theano_shared=True, validation=True):
    ''' Loads the SVHN dataset
    :type ds_rate: float
    :param ds_rate: downsample rate; should be larger than 1, if provided.
    :type theano_shared: boolean
    :param theano_shared: If true, the function returns the dataset as Theano
    shared variables. Otherwise, the function returns raw data.
    :type validation: boolean
    :param validation: If true, extract validation dataset from train dataset.
    '''
    if ds_rate is not None:
        assert(ds_rate > 1.)

    # Download the CIFAR-10 dataset if it is not present
    def check_dataset(dataset):
        # Check if dataset is in the data directory.
        new_path = os.path.join(
            os.path.split(__file__)[0],
            "..",
            "data",
            dataset
        )
        #f_name = new_path.replace("src/../data/%s"%dataset, "data/") 
        f_name = os.path.join(
            os.path.split(__file__)[0],
            "..",
            "data"
        )
        if (not os.path.isfile(new_path)):
            from six.moves import urllib
            origin = (
                'https://www.cs.toronto.edu/~kriz/' + dataset
            )
            print('Downloading data from %s' % origin)
            urllib.request.urlretrieve(origin, new_path) 
             
        tar = tarfile.open(new_path)
        file_names = tar.getnames()
        for file_name in file_names:
            tar.extract(file_name,f_name)
        tar.close()              
        
        return f_name
    
    f_name=check_dataset('cifar-10-matlab.tar.gz')
    
    train_batches=os.path.join(f_name,'cifar-10-batches-mat/data_batch_1.mat')
    
    
    # Load data and convert data format
    train_batches=['data_batch_1.mat','data_batch_2.mat','data_batch_3.mat','data_batch_4.mat','data_batch_5.mat']
    train_batch=os.path.join(f_name,'cifar-10-batches-mat',train_batches[0])
    train_set=scipy.io.loadmat(train_batch)
    train_set['data']=train_set['data']/255.
    for i in range(4):
        train_batch=os.path.join(f_name,'cifar-10-batches-mat',train_batches[i+1])
        temp=scipy.io.loadmat(train_batch)
        train_set['data']=numpy.concatenate((train_set['data'],temp['data']/255.),axis=0)
        train_set['labels']=numpy.concatenate((train_set['labels'].flatten(),temp['labels'].flatten()),axis=0)
    
    test_batches=os.path.join(f_name,'cifar-10-batches-mat/test_batch.mat')
    test_set=scipy.io.loadmat(test_batches)
    test_set['data']=test_set['data']/255.
    test_set['labels']=test_set['labels'].flatten()
    
    train_set=(train_set['data'],train_set['labels'])
    test_set=(test_set['data'],test_set['labels'])
    

    # Downsample the training dataset if specified
    train_set_len = len(train_set[1])
    if ds_rate is not None:
        train_set_len = int(train_set_len // ds_rate)
        train_set = [x[:train_set_len] for x in train_set]

    if validation:
        # Extract validation dataset from train dataset
        valid_set = [x[-(train_set_len//5):] for x in train_set]
        train_set = [x[:-(train_set_len//5)] for x in train_set]

        # train_set, valid_set, test_set format: tuple(input, target)
        # input is a numpy.ndarray of 2 dimensions (a matrix)
        # where each row corresponds to an example. target is a
        # numpy.ndarray of 1 dimension (vector) that has the same length as
        # the number of rows in the input. It should give the target
        # to the example with the same index in the input.

        if theano_shared:
            test_set_x, test_set_y = shared_dataset(test_set)
            valid_set_x, valid_set_y = shared_dataset(valid_set)
            train_set_x, train_set_y = shared_dataset(train_set)

            rval = [(train_set_x, train_set_y), (valid_set_x, valid_set_y),
                (test_set_x, test_set_y)]
        else:
            rval = [train_set, valid_set, test_set]

        return rval
    
    else:

        if theano_shared:
            test_set_x, test_set_y = shared_dataset(test_set)
            train_set_x, train_set_y = shared_dataset(train_set)

            rval = [(train_set_x, train_set_y), (test_set_x, test_set_y)]
        else:
            rval = [train_set, test_set]

        return rval

def RMSprop(cost, params, lr=0.01, rho=0.5, epsilon=1e-6):
    grads = T.grad(cost=cost, wrt=params)
    updates = []
    for p, g in zip(params, grads):
        acc = theano.shared(p.get_value() * 0.)
        acc_new = rho * acc + (1 - rho) * g ** 2
        gradient_scaling = T.sqrt(acc_new + epsilon)
        g = g / gradient_scaling
        updates.append((acc, acc_new))
        updates.append((p, p - lr * g))
        #print(acc.eval())
        #print(p.eval())
    return updates

def Momentum(cost, params, eps = 0.01, alpha = 0.9):
    grads = T.grad(cost=cost,wrt=params)
    updates = []
    v = [theano.shared(numpy.zeros(param_i.shape.eval(), dtype=theano.config.floatX),borrow=True) for param_i in params]
    for param_i, grad_i, v_i in zip(params, grads, v):
        v_next = alpha*v_i - eps*grad_i
        updates.append((v_i, v_next))
        updates.append((param_i, param_i + v_next))
    return updates
    
def MomentumWithDecay(cost, params, itr, lr_base = 0.01, lr_decay = 1.0, lr_min = 0.00001, momentum = 0.9):
    grads = T.grad(cost=cost,wrt=params)
    lr = T.max(lr_base * lr_decay**itr, lr_min)
    updates = []
    v = [theano.shared(numpy.zeros(param_i.shape.eval(), dtype=theano.config.floatX),borrow=True) for param_i in params]
    for param_i, grad_i, v_i in zip(params, grads, v):
        v_next = momentum*v_i - lr*grad_i
        updates.append((v_i, v_next))
        updates.append((param_i, param_i + v_next))
    return updates

def MomentumWithMultiStepDecay(cost, params, itr, lr_base = 0.025, lr_decay = 0.1, momentum = 0.9, step_values = []): 
    grads = T.grad(cost=cost,wrt=params)  
    updates = []

    step = theano.shared(numpy.cast[theano.config.floatX](0))
    flg = sum([T.eq(itr, step_value) for step_value in step_values]) # check if itr is one of the step values
    step_next = T.switch(flg, step + numpy.cast[theano.config.floatX](1), step)
    updates.append((step,step_next))
    lr = lr_base*lr_decay**step_next

    v = [theano.shared(numpy.zeros(param_i.shape.eval(), dtype=theano.config.floatX),borrow=True) for param_i in params]
    for param_i, grad_i, v_i in zip(params, grads, v):
        v_next = momentum*v_i - lr*grad_i #theano.clone(grad_i, replace={param_i: param_i + momentum*v_i})
        updates.append((v_i, v_next))
        updates.append((param_i, param_i + v_next))
    return updates

def MomentumG(cost, params, itr, lr_base = 0.01, lr_decay = 1.0, lr_min = 0.00001, momentum = 0.9):
    lr = T.max(lr_base*numpy.exp(-lr_decay**(itr-1)), lr_min)
    momentum =theano.shared(numpy.cast[theano.config.floatX](momentum), name='momentum')
    updates = []
    for param in params:
        param_update = theano.shared(param.get_value()*numpy.cast[theano.config.floatX](0.))    
        updates.append((param, param - lr*param_update))
        updates.append((param_update, momentum*param_update + (numpy.cast[theano.config.floatX](1.) - momentum)*T.grad(cost, param)))
    return updates
