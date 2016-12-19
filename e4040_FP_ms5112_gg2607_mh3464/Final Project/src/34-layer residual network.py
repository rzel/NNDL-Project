from project_nn import load_data_SVHN
def ResidualNetwork_34(decay = 0.5, momentum = 0.9, learning_rate=0.001, n_epochs = 400, batch_size = 500):
    rng = numpy.random.RandomState(5112)

    data = load_data_SVHN(ds_rate=None,theano_shared=True, validation=False)
    train_set_x, train_set_y = data[0]
    test_set_x, test_set_y = data[1]   
    
    n_train_batches = train_set_x.get_value(borrow=True).shape[0]
    n_test_batches = test_set_x.get_value(borrow=True).shape[0]

    n_train_batches //= batch_size
    n_test_batches //= batch_size
     
    index = T.lscalar()
    x = T.matrix('x')
    y = T.ivector('y')
    
    print('... building model')
    d = train_set_x.get_value(borrow=True).shape[1]
    allLayers = []
    layer0_input = x.reshape((batch_size,1,d,d)) # For MNIST

    layer0 = LeNetConvPoolLayer(
        rng,
        input=layer0_input,
        image_shape=(batch_size,1,d,d),
        filter_shape=(64,1,7,7),
        poolsize=(2,2),
        padborder=True
    )
    allLayers.append(layer0)
    
    layer1_output = T.signal.pool.pool_2d(layer0.output,ds=(2,2),ignore_border=True,st=(2,2),mode='max')

    layer2 = LeNetConvPoolLayer(
        rng,
        input= layer1_output,
        image_shape=(batch_size,64,d,d),
        filter_shape=(64,64,3,3),
        poolsize=(1,1),
        padborder=True
    )
    allLayers.append(layer2)
    
    layer3 = LeNetConvPoolLayer(
        rng,
        input= layer2.output,
        image_shape=(batch_size,64,d,d),
        filter_shape=(64,64,3,3),
        poolsize=(1,1),
        padborder=True
    )
    allLayers.append(layer3)
    
    layer4 = LeNetConvPoolLayer(
        rng,
        input= layer3.output+layer1_output,
        image_shape=(batch_size,64,d,d),
        filter_shape=(64,64,3,3),
        poolsize=(1,1),
        padborder=True
    )
    allLayers.append(layer4)
    
    layer5 = LeNetConvPoolLayer(
        rng,
        input= layer4.output,
        image_shape=(batch_size,64,d,d),
        filter_shape=(64,64,3,3),
        poolsize=(1,1),
        padborder=True
    )
    allLayers.append(layer5)
    
    layer6 = LeNetConvPoolLayer(
        rng,
        input= layer5.output+layer3.output,
        image_shape=(batch_size,64,d,d),
        filter_shape=(64,64,3,3),
        poolsize=(1,1),
        padborder=True
    )
    allLayers.append(layer6)
    
    layer7 = LeNetConvPoolLayer(
        rng,
        input= layer6.output,
        image_shape=(batch_size,64,d,d),
        filter_shape=(64,64,3,3),
        poolsize=(1,1),
        padborder=True
    )
    allLayers.append(layer7)
    
    layer8 = LeNetConvPoolLayer(
        rng,
        input= layer7.output+layer5.output,  
        image_shape=(batch_size,64,d,d),
        filter_shape=(128,64,3,3),
        poolsize=(2,2),
        padborder=True
    )
    allLayers.append(layer8)
    
    layer9 = LeNetConvPoolLayer(
        rng,
        input= layer8.output,
        image_shape=(batch_size,128,d,d),
        filter_shape=(128,128,3,3),
        poolsize=(1,1),
        padborder=True
    )
    allLayers.append(layer9)
   
    layer10 = LeNetConvPoolLayer(
        rng,
        input= layer9.output + layer7.output,  # THIS IS NOT GOING TO WORK, NEED TO CHANGE DIMENSION OF LAYER5.OUTPUT
        image_shape=(batch_size,128,d,d),
        filter_shape=(128,128,3,3),
        poolsize=(1,1),
        padborder=True
    )
    allLayers.append(layer10)
   
    layer11 = LeNetConvPoolLayer(
        rng,
        input= layer10.output,
        image_shape=(batch_size,128,d,d),
        filter_shape=(128,128,3,3),
        poolsize=(1,1),
        padborder=True
    )
    allLayers.append(layer11)

    layer12 = LeNetConvPoolLayer(
        rng,
        input= layer11.output+layer9.output,
        image_shape=(batch_size,128,d,d),
        filter_shape=(128,128,3,3),
        poolsize=(1,1),
        padborder=True
    )    
    allLayers.append(layer12)
    
    layer13 = LeNetConvPoolLayer(
        rng,
        input= layer12.output,
        image_shape=(batch_size,128,d,d),
        filter_shape=(128,128,3,3),
        poolsize=(1,1),
        padborder=True
    )    
    allLayers.append(layer13)
    
    layer14 = LeNetConvPoolLayer(
        rng,
        input= layer13.output+layer11.output,
        image_shape=(batch_size,128,d,d),
        filter_shape=(128,128,3,3),
        poolsize=(1,1),
        padborder=True
    )
    allLayers.append(layer14)
    
    layer15 = LeNetConvPoolLayer(
        rng,
        input= layer14.output,
        image_shape=(batch_size,128,d,d),
        filter_shape=(128,128,3,3),
        poolsize=(1,1),
        padborder=True
    )
    allLayers.append(layer15)
    
    layer16 = LeNetConvPoolLayer(
        rng,
        input= layer15.output+layer13.output,
        image_shape=(batch_size,128,d,d),
        filter_shape=(256,128,3,3),
        poolsize=(2,2),
        padborder=True
    )
    allLayers.append(layer16)
    
    layer17 = LeNetConvPoolLayer(
        rng,
        input= layer16.output,
        image_shape=(batch_size,256,d,d),
        filter_shape=(256,256,3,3),
        poolsize=(1,1),
        padborder=True
    ) 
    allLayers.append(layer17)
    
    layer18 = LeNetConvPoolLayer(
        rng,
        input= layer17.output+layer15.output, # THIS IS NOT GOING TO WORK, NEED TO ADD ZEROS TO LAYER15.OUTPUT
        image_shape=(batch_size,256,d,d),
        filter_shape=(256,256,3,3),
        poolsize=(1,1),
        padborder=True
    )
    allLayers.append(layer18)
    
    layer19 = LeNetConvPoolLayer(
        rng,
        input= layer18.output, 
        image_shape=(batch_size,256,d,d),
        filter_shape=(256,256,3,3),
        poolsize=(1,1),
        padborder=True
    ) 
    allLayers.append(layer19)
    
    layer20 = LeNetConvPoolLayer(
        rng,
        input= layer19.output+layer17.output, 
        image_shape=(batch_size,256,d,d),
        filter_shape=(256,256,3,3),
        poolsize=(1,1),
        padborder=True
    )
    allLayers.append(layer20)
    
    layer21 = LeNetConvPoolLayer(
        rng,
        input= layer20.output, 
        image_shape=(batch_size,256,d,d),
        filter_shape=(256,256,3,3),
        poolsize=(1,1),
        padborder=True
    ) 
    allLayers.append(layer21)
    
    layer22 = LeNetConvPoolLayer(
        rng,
        input= layer21.output+layer19.output, 
        image_shape=(batch_size,256,d,d),
        filter_shape=(256,256,3,3),
        poolsize=(1,1),
        padborder=True
    ) 
    allLayers.append(layer22)
    
    layer23 = LeNetConvPoolLayer(
        rng,
        input= layer22.output, 
        image_shape=(batch_size,256,d,d),
        filter_shape=(256,256,3,3),
        poolsize=(1,1),
        padborder=True
    ) 
    allLayers.append(layer23)
    
    layer24 = LeNetConvPoolLayer(
        rng,
        input= layer23.output+layer21.output, 
        image_shape=(batch_size,256,d,d),
        filter_shape=(256,256,3,3),
        poolsize=(1,1),
        padborder=True
    ) 
    allLayers.append(layer24)
    
    layer25 = LeNetConvPoolLayer(
        rng,
        input= layer24.output, 
        image_shape=(batch_size,256,d,d),
        filter_shape=(256,256,3,3),
        poolsize=(1,1),
        padborder=True
    ) 
    allLayers.append(layer25)
    
    layer26 = LeNetConvPoolLayer(
        rng,
        input= layer25.output+layer23.output, 
        image_shape=(batch_size,256,d,d),
        filter_shape=(256,256,3,3),
        poolsize=(1,1),
        padborder=True
    ) 
    allLayers.append(layer26)
    
    layer27 = LeNetConvPoolLayer(
        rng,
        input= layer26.output, 
        image_shape=(batch_size,256,d,d),
        filter_shape=(256,256,3,3),
        poolsize=(1,1),
        padborder=True
    ) 
    allLayers.append(layer27)
    
    layer28 = LeNetConvPoolLayer(
        rng,
        input= layer27.output+layer25.output, 
        image_shape=(batch_size,256,d,d),
        filter_shape=(512,256,3,3),
        poolsize=(2,2),
        padborder=True
    ) 
    allLayers.append(layer28)

    layer29 = LeNetConvPoolLayer(
        rng,
        input= layer28.output, 
        image_shape=(batch_size,512,d,d),
        filter_shape=(512,512,3,3),
        poolsize=(1,1),
        padborder=True
    ) 
    allLayers.append(layer29)
    
    layer30 = LeNetConvPoolLayer(
        rng,
        input= layer29.output+layer27.output, # THIS IS NOT GOING TO WORK, NEED TO ADD ZEROS TO LAYER27.OUTPUT 
        image_shape=(batch_size,512,d,d),
        filter_shape=(512,512,3,3),
        poolsize=(1,1),
        padborder=True
    ) 
    allLayers.append(layer30)
    
    layer31 = LeNetConvPoolLayer(
        rng,
        input= layer30.output, 
        image_shape=(batch_size,512,d,d),
        filter_shape=(512,512,3,3),
        poolsize=(1,1),
        padborder=True
    ) 
    allLayers.append(layer31)
    
    layer32 = LeNetConvPoolLayer(
        rng,
        input= layer31.output+layer29.output, 
        image_shape=(batch_size,512,d,d),
        filter_shape=(512,512,3,3),
        poolsize=(1,1),
        padborder=True
    ) 
    allLayers.append(layer32)
    
    layer33 = LeNetConvPoolLayer(
        rng,
        input = layer32.output,
        image_shape=(batch_size,512,d,d),
        filter_shape=(512,512,3,3),
        poolsize=(1,1),
        padborder=True
    )
    allLayers.append(layer33)
    
    layer34_output = T.signal.pool.pool_2d(layer33.output+layer31.output,ds=(1,1),ignore_border=True,
                                           st=(1,1),mode='average_inc_pad')
    
    layerLL = LogisticRegression(
        input=layer34_output,
        n_in = 256,
        n_out = 10
    )
    allLayers.append(layerLL)
        
    cost = layerLL.negative_log_likelihood(y)
    
    test_model = theano.function(
        [index],
        layerLL.errors(y),
        givens={
            x: test_set_x[index*batch_size:(index+1)*batch_size],
            y: test_set_y[index*batch_size:(index+1)*batch_size],
            training_enabled: numpy.cast['int32'](0)
        }
    )
    
    validate_model = theano.function(
        [index],
        layerLL.errors(y),
        givens={
            x: valid_set_x[index*batch_size:(index+1)*batch_size],
            y: valid_set_y[index*batch_size:(index+1)*batch_size],
            training_enabled: numpy.cast['int32'](0)
        }
    )
    
    params = sum([x.params for x in allLayers], [])
    
    # RMSprop
    updates = RMSprop(cost,params,lr=learning_rate,rho=rho)
    
    train_model = theano.function(
        [index],
        cost,
        updates=updates,
        givens={
            x: train_set_x[index*batch_size:(index+1)*batch_size],
            y: train_set_y[index*batch_size:(index+1)*batch_size],
            training_enabled: numpy.cast['int32'](1)
        }
    )
    
    print('... training')
    # early-stopping parameters
    patience = 10000  # look as this many examples regardless
    patience_increase = 2  # wait this much longer when a new best is
                           # found
    improvement_threshold = 0.995  # a relative improvement of this much is
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

    while (epoch < n_epochs) and (not done_looping):
        epoch = epoch + 1
        for minibatch_index in range(n_train_batches):

            iter = (epoch - 1) * n_train_batches + minibatch_index
            cost_ij = train_model(minibatch_index)

            if (iter + 1) % validation_frequency == 0:

                # compute zero-one loss on validation set
                validation_losses = [validate_model(i) for i
                                     in range(n_valid_batches)]
                this_validation_loss = numpy.mean(validation_losses)
                print('epoch %i, minibatch %i/%i, validation error %f %%' %
                      (epoch, minibatch_index + 1, n_train_batches,
                       this_validation_loss * 100.))

                # if we got the best validation score until now
                if this_validation_loss < best_validation_loss:

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
                    print(('     epoch %i, minibatch %i/%i, test error of '
                           'best model %f %%') %
                          (epoch, minibatch_index + 1, n_train_batches,
                           test_score * 100.))

            if patience <= iter:
                done_looping = True
                break

    end_time = timeit.default_timer()
    print('Optimization complete.')
    print('Best validation score of %f %% obtained at iteration %i, '
          'with test performance %f %%' %
          (best_validation_loss * 100., best_iter + 1, test_score * 100.))
    print(('The code for file ' +
           os.path.split(__file__)[1] +
           ' ran for %.2fm' % ((end_time - start_time) / 60.)), file=sys.stderr)
    

    