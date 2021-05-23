def build_metric_network(single_embedding_shape):
    '''
    Define the neural network to learn the metric
    Input :
            single_embedding_shape : shape of input embeddings or feature map. Must be an array

    '''
    # compute shape for input
    input_shape = single_embedding_shape
    # the two input embeddings will be concatenated
    input_shape[0] = input_shape[0] * 2

    # Neural Network
    network = Sequential(name="learned_metric")
    network.add(Dense(10, activation='relu',
                      input_shape=input_shape,
                      kernel_regularizer=l2(1e-3),
                      kernel_initializer='he_uniform'))
    network.add(Dense(10, activation='relu',
                      kernel_regularizer=l2(1e-3),
                      kernel_initializer='he_uniform'))
    network.add(Dense(10, activation='relu',
                      kernel_regularizer=l2(1e-3),
                      kernel_initializer='he_uniform'))

    # Last layer : binary softmax
    network.add(Dense(2, activation='softmax'))

    # Select only one output value from the softmax
    network.add(Lambda(lambda x: x[:, 0]))

    return network

#modele en 4xloss
network4 = build_network(input_shape,embeddingsize=10)
network4.set_weights(network3.get_weights()) #copy weights to have identical networks
metric_network4 = build_metric_network(single_embedding_shape=[10])
network4_train = build_model4(input_shape,network4,metric_network4,margin=alpha1, margin2=alpha2)
optimizer4 = Adam(lr = 0.00006)
network4_train.compile(loss=None,optimizer=optimizer4)
network4_train.summary()
plot_model(network4_train,show_shapes=True, show_layer_names=True, to_file=project_path+'model_summary_4x.png')

n_iteration=0

quadruplets = get_batch_random(2,dataset_train)
print("Checking batch width, should be 4 : ",len(quadruplets))
print("Shapes in the batch A:{0} P:{1} N:{2} N2:{3}".format(quadruplets[0].shape, quadruplets[1].shape, quadruplets[2].shape, quadruplets[3].shape))
drawQuadriplets(quadruplets)
hardtriplets,hardquadruplets = get_batch_hard(50,1,1,network3,network4,metric_network4,dataset_train)
print("Shapes in the hardbatch 4x A:{0} P:{1} N:{2}".format(hardquadruplets[0].shape, hardquadruplets[1].shape, hardquadruplets[2].shape, hardquadruplets[3].shape))
drawQuadriplets(hardquadruplets)

print("Starting training process!")
print("-------------------------------------")
t_start = time.time()
for i in range(1, n_iter+1):
    microtask_start =time.time()
    triplets,quadruplets = get_batch_hardOptimized(100,16,16,network3,network4,metric_network4, dataset_train)
    timetogetbatch = time.time()-microtask_start
    microtask_start =time.time()
    loss1 = network3_train.train_on_batch(triplets, None)
    timebatch3 = time.time()-microtask_start
    microtask_start =time.time()
    loss2 = network4_train.train_on_batch(quadruplets, None)
    timebatch4 = time.time()-microtask_start
    n_iteration += 1
    if i % log_every == 0:
        wandb.log({'loss3x': loss1,'loss4x': loss2}, step=n_iteration)
    if i % evaluate_every == 0:
        elapsed_minutes = (time.time()-t_start)/60.0
        rate = i/elapsed_minutes
        eta = datetime.now()+timedelta(minutes=(n_iter-i)/rate)
        eta = eta + timedelta(hours=0) #french time
        print("[{4}] iteration {0}: {1:.1f} iter/min, Train Loss: {2} {3}, eta : {5}".format(i, rate,loss1,loss2,n_iteration,eta.strftime("%Y-%m-%d %H:%M:%S") ))
        network3_train.save_weights('{1}3x-temp_weights_{0:08d}.h5'.format(n_iteration,model_path))
        network4_train.save_weights('{1}4x-temp_weights_{0:08d}.h5'.format(n_iteration,model_path))
#Final save
network3_train.save_weights('{1}3x-temp_weights_{0:08d}.h5'.format(n_iteration,model_path))
network4_train.save_weights('{1}4x-temp_weights_{0:08d}.h5'.format(n_iteration,model_path))
print("Done !")
