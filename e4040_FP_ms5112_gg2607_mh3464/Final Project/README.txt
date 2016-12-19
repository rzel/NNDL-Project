####################################################
     Highway Networks
     E4040.2016Fall.DLGS.report
####################################################

PYTHON CODES
-------------
* 34-layer residual network.py
  (This file contains a function for the 34 layer residual networks of Figure 3 in He et al[4].)
	- ResidualNetwork_34: function that produces a 34 layer residual network.

* ConvHighwayNetworks.py
  (This file was used to test Highway 1-2 and ResNets)
	- ResNet: class that represents residual networks with 6n+2 layers for Table 6 in He et al[4].
	- ConvHighway1: class that represents Highway 1 for Table 1 in Srivastava et al [2].
	- ConvHighway2: class that represents Highway 2 for Table 1 in Srivastava et al [2].
	- test_ConvHighway: function that trains Highway 1-2 or ResNet on CIFAR-10 datasets.
	
* HighwayNetworks.py
  (This file was used to do hyperparameter search for highway networks)
	- test_Highway: function that trains Highway Network on MNIST datasets using RMSprop.
	- test_Highway_Momentum: function that trains Highway Network on MNIST datasets using SGD with momentum.

* HighwayNetworksOutput.py
  (This file was used to produce the heat maps for Figure 2 in Srivastava et al [2].)
	- test_Highway_Momentum_output: function that trains Highway Network on MNIST datasets using SGD with momentum.

* project_nn.py
  (This file contains all the building blocks used by HighwayNetworks.py)
	- LogisticRegression: class that represents a fully-connected softmax layer
	- HiddenLayer: class that represents a fully-connected layer
	- HighwayLayer: class that represents a highway layer
	- HighwayNetwork: class that represents a highway layer
	- myMLP: class that represents a multi-layer perceptron
	- drop: function that performs dropout
	- DropoutHiddenLayer: class that represents a fully-connected layer with dropout
	- train_result: class that contains all the training results  
	- train_nn: function that trains neural network models
	- load_data: function that loads MNIST data
	- RMSprop: function that generates the update rule for RMSprop algorithm. 
	- Momentum: function that generates the update rule for the standard SGD with momentum algorithm 
	- MomentumWithDecay: function that generates the update rule for the standard SGD with momentum and decay 
	- MomentumG: function that generates the update rule for another type of SGD with momentum and decay

* project_nn_output.py
  (This file contains all the building blocks used by HighwayNetworksOutput.py. 
   It is almost identical to project_nn.py but has some additional codes to create the heatmaps.)
	- LogisticRegression: class that represents a fully-connected softmax layer
	- HiddenLayer: class that represents a fully-connected layer
	- HighwayLayer: class that represents a highway layer
	- HighwayNetwork: class that represents a highway layer
	- myMLP: class that represents a multi-layer perceptron
	- drop: function that performs dropout
	- DropoutHiddenLayer: class that represents a fully-connected layer with dropout
	- train_result: class that contains all the training results  
	- train_nn: function that trains neural network models
	- load_data: function that loads MNIST data
	- RMSprop: function that generates the update rule for RMSprop algorithm. 
	- Momentum: function that generates the update rule for the standard SGD with momentum algorithm 
	- MomentumWithDecay: function that generates the update rule for the standard SGD with momentum and decay 
	- MomentumG: function that generates the update rule for another type of SGD with momentum and decay
	
* project_nn_resnet.py
  (This file contains all the building blocks used by ConvHighwayNetworks.py)
	- LogisticRegression: class that represents a fully-connected softmax layer
	- HiddenLayer: class that represents a fully-connected layer
	- ConvLayer: class that represents a convolutional layer
	- HighwayLayer: class that represents a highway layer
	- HighwayNetwork: class that represents a highway layer
	- myMLP: class that represents a multi-layer perceptron
	- ActivationLayer: class that represents an activation layer (only applying activation function)
	- drop: function that performs dropout
	- DropoutLayer: class that represents a dropout layer
	- PoolingLayer: class that represents a pooling layer
	- DropoutHiddenLayer: class that represents a fully-connected layer with dropout
	- train_result: class that contains all the training results  
	- doNothing: dummy function for using no data augmentation
	- train_nn: function that trains neural network models
	- train_nn_NoValidation: function that trains neural network models without validation set
	- load_data: function that loads MNIST data
	- shared_dataset: function that loads the dataset into shared variables
	- load_data_SVHN: function that loads CIFAR-10 data
	- RMSprop: function that generates the update rule for RMSprop algorithm. 
	- Momentum: function that generates the update rule for the standard SGD with momentum algorithm 
	- MomentumWithDecay: function that generates the update rule for the standard SGD with momentum and decay 
	- MomentumWithMultiStepDecay: function that generates the update rule for the standard SGD with momentum and multi-step decay
	- MomentumG: function that generates the update rule for another type of SGD with momentum and decay
	
* project_utils.py
  (This file contains utility functions to test Highway 1-2 and ResNets)
	- translate_image: function that randomly translates CIFAR-10 images for data augmentation
	- rotate_image: function that randomly rotates CIFAR-10 images for data augmentation
	- flip_image: function that randomly flips CIFAR-10 images horizontally for data augmentation
	- zero_padding: function that adds extra zero entries when the number of channels increases for ResNets

* test_codes.py 
 (This file only contains a set of execution codes that we used to train Highway 1-2 and Resnets) 

REFERENCES
-------------
[2] R.K. Srivastava, K. Greff, J. Schmidhuber, "Highway Networks", 
3 November 2015, ICML 2015 Deep Learning workshop,

[4] K. He, X. Zhang, S. Ren, J. Sun, "Deep Residual Learning for Image Recognition", 
10 December 2015, Tech report