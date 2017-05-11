from model import * 
from DataHandler import * 
from Logger import * 
from collections import OrderedDict 

# hyperparameters / parameters
params = OrderedDict()
params['batch_size'] = 64
params['initial_eta'] = 3e-4
params['load_weights'] = False
params['optimizer'] = 'rmsprop'
params['num_gen_units'] = 128 # num channels for second last layer output. 
params['z_dim'] = 512
params['disc_iter'] = 1
params['gen_iter'] = 1
params['image_prepro'] = 'DCGAN' # ( /250.; -0.5; /0.5) taken from DCGAN repo.

generator_layers = generator(num_units=params['num_gen_units'], z_dim=params['z_dim'])
discriminator_layers = discriminator()
generator = generator_layers[-1]
critic = discriminator_layers[-1]


dh = DataHandler()
eh = ExpHandler(params)

# placeholders 
images = T.tensor4('images from dataset')
index = T.lscalar() # index to a [mini]batch
eta = theano.shared(lasagne.utils.floatX(params['initial_eta']))
a, b, c = 0, 1, 1

print 'initializing functions'
gen_output = ll.get_output(generator)
gen_params = ll.get_all_params(generator, trainable=True)
critic_params = ll.get_all_params(critic, trainable=True)
real_out = ll.get_output(critic, inputs=images)
fake_out = ll.get_output(critic, inputs=gen_output)
 
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
                               outputs=[gen_output], 
                               name='test_gen')

'''
training section 
'''
print 'staring training'
for epoch in range(3000000):
    gen_err = 0
    disc_err = 0

    for _ in range(50):
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
    eh.save_image(test_gen()[0])
    eh.end_of_epoch()

    print epoch
    print("gen  loss:\t\t{}".format(gen_err / 50. ))
    print("disc loss:\t\t{}".format(disc_err / 50. ))
 
