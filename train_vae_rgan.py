from model import * 
from DataHandler import * 
from ExpHandler import * 
from collections import OrderedDict 

# hyperparameters / parameters
params = OrderedDict()
params['batch_size'] = 4
params['initial_eta'] = 2e-4
params['load_weights'] = (67, 2150) # version/epoch tupple pair None
params['optimizer'] = 'rmsprop'
params['num_gen_units'] = 128 # num channels for second last layer output. 
params['z_dim'] = 512
params['disc_iter'] = 1
params['gen_iter'] = 1
params['epoch_iter'] = 0
params['seq_length'] = 64
params['skip_frames'] = 1
params['comments'] = 'testing longer sequences with RGAN'
params['image_prepro'] = 'DCGAN' # ( /250.; -0.5; /0.5) taken from DCGAN repo.

generator_layers = generator(num_units=params['num_gen_units'], z_dim=params['z_dim'])
discriminator_layers = discriminator_3D(seq_length=(params['seq_length']/params['skip_frames']))
encoder_layers = encoder()
recurrent_layers = recurrent()
generator = generator_layers[-1]
critic = discriminator_layers[-1]
encoder = encoder_layers[-1]
recurrent = recurrent_layers[-1]

dh = DataHandler(time_len=params['seq_length'], 
                 skip_frames=params['skip_frames'], 
                 num_batches=params['epoch_iter']+1)
eh = ExpHandler(params)

# placeholders 
images = T.tensor5('images from dataset')
index = T.lscalar() # index to a [mini]batch
eta = theano.shared(lasagne.utils.floatX(params['initial_eta']))
a, b, c = 0, 1, 1

enc_output = ll.get_output(encoder, inputs=images)
# we need to fetch input at each timestep for t=0,...,'seq_length'
rec_outputs = []
gen_outputs = []
rec_outputs.append(ll.get_output(recurrent)) # timestep t
for i in range(params['seq_length'] / params['skip_frames']):
    # timestep t
    gen_outputs.append(ll.get_output(generator, inputs=rec_outputs[-1]))
    # timestep t+1
    rec_outputs.append(ll.get_output(recurrent, inputs=rec_outputs[-1]))

# next we need to merge all the images together 
critic_input = T.stack(gen_outputs, axis=1)

gen_params = ll.get_all_params([generator, recurrent], trainable=True)
critic_params = ll.get_all_params(critic, trainable=True)
rec_params = ll.get_all_params(recurrent, trainable=True)
real_out = ll.get_output(critic, inputs=images)
fake_out = ll.get_output(critic, inputs=critic_input)
 
# losses 
critic_loss = (lasagne.objectives.squared_error(real_out, b).mean() +
 		 lasagne.objectives.squared_error(fake_out, a).mean())         
gen_loss = lasagne.objectives.squared_error(fake_out, c).mean()

gen_grads = theano.grad(gen_loss, wrt=gen_params)
critic_grads = theano.grad(critic_loss, wrt=critic_params)
gen_grads_norm = sum(T.sum(T.square(grad)) for grad in gen_grads) / len(gen_grads)
critic_grads_norm = sum(T.sum(T.square(grad)) for grad in critic_grads) / len(critic_grads)

# updates
gen_updates= optimizer_factory(
    params['optimizer'], gen_grads, gen_params, params['initial_eta'])
critic_updates = optimizer_factory(
    params['optimizer'], critic_grads, critic_params, params['initial_eta'])

# function outputs
critic_fn_output = OrderedDict()
critic_fn_output['real_out_mean'] = (real_out).mean()
critic_fn_output['fake_out_mean'] = (fake_out).mean()
critic_fn_output['loss'] = critic_loss
critic_fn_output['grads_norm'] = critic_grads_norm

gen_fn_output = OrderedDict()
gen_fn_output['fake_out_mean'] = (fake_out).mean()
gen_fn_output['loss'] = gen_loss
gen_fn_output['grads_norm'] = gen_grads_norm

eh.add_model('gen', generator_layers,  gen_fn_output)
eh.add_model('disc', discriminator_layers, critic_fn_output)

# functions
train_critic = theano.function(inputs=[index], 
                               outputs=critic_fn_output.values(), 
                               updates=critic_updates,
                               givens={images: dh.GPU_image[index * params['batch_size']: 
                                                         (index+1) * params['batch_size']]},
                               name='train_critic')
train_gen =    theano.function(inputs=[],
                               outputs=gen_fn_output.values(),
                               updates = gen_updates,
                               name='train_gen')
test_gen =     theano.function(inputs=[],
                               outputs=[critic_input], 
                               name='test_gen')

'''
training section 
'''
print 'staring training'
for epoch in range(3000000):
    gen_err = 0
    disc_err = 0

    for _ in range(params['epoch_iter']):
        batch_no = dh.get_next_batch_no()
        for _ in range (params['gen_iter']) :
            gen_out = np.array(train_gen())
            gen_err += gen_out
            eh.record('gen', gen_out)
        for _ in range (params['disc_iter']) :
            disc_out = np.array(train_critic(batch_no))
            disc_err += disc_out
            eh.record('disc', disc_out)

    # test out model
    frames = extract_video_frames(test_gen()[0])
    real_frames = extract_video_frames(dh.GPU_image.get_value()[batch_no * params['batch_size']: 
                                        (batch_no+1) * params['batch_size']])
    eh.save_image(frames)
    eh.save_image(real_frames, real_img=True)
    eh.end_of_epoch()

    print epoch
    '''
    print("gen  loss:\t\t{}".format(gen_err / ( params['epoch_iter'] * params['gen_iter'])))
    print("disc loss:\t\t{}".format(disc_err / ( params['epoch_iter'] * params['disc_iter'])))
    '''
