from ConvHighwayNetworks import test_ConvHighway
from project_nn import load_data_SVHN
import sys
import os

# stop buffering
sys.stdout = os.fdopen(sys.stdout.fileno(), 'w', 0)

data = load_data_SVHN(ds_rate=None,theano_shared=True, validation=False)
test_ConvHighway(data, model=1, learning_rate=0.025, lr_decay=0.1, momentum=0.9, step_values = [30000, 100000, 150000, 175000], 
                 n_epochs=400, b_T=-2, drop_rate=0.2, batch_size=100, verbose=True)
