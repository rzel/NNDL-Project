############################################
# Test Code for Highway Network 1&2 and Residual Network 1&2
############################################

data = load_data_SVHN(ds_rate=None,theano_shared=True, validation=False)

## Highway 1
test_ConvHighway(data, model=1, learning_rate=0.025, lr_decay=0.1, momentum=0.9, step_values = [100000, 150000, 175000], 
                 n_epochs=400, b_T=-2, drop_rate=0.2, batch_size=100, verbose=True)
				 
				 
## Highway 2
# used different step_values
test_ConvHighway(data, model=2, learning_rate=0.025, lr_decay=0.316, momentum=0.9, step_values = [30000, 100000, 130000, 150000, 165000, 175000], 
                 n_epochs=400, b_T=-2, drop_rate=0.2, batch_size=100, verbose=True)			 
				 
## ResNet 1
test_ConvHighway(data, model=11, learning_rate=0.025, lr_decay=0.1, momentum=0.9, step_values = [100000, 150000, 175000], 
                 n_epochs=400, drop_rate=0.2, batch_size=100, verbose=True)
		 

## ResNet 2
test_ConvHighway(data, model=12, learning_rate=0.025, lr_decay=0.1, momentum=0.9, step_values = [100000, 150000, 175000], 
                 n_epochs=400, drop_rate=0.2, batch_size=100, verbose=True)