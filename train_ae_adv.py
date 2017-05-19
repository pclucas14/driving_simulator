from model import * 
from DataHandler import * 
from ExpHandler import * 
from collections import OrderedDict 

# hyperparameters / parameters
params = OrderedDict()
params['batch_size'] = 64
params['initial_eta'] = 2e-4
params['critic_eta'] = 2e-4
params['load_weights'] = (12, 1000)#None# version/epoch tupple pair
params['optimizer'] = 'rmsprop'
params['num_gen_units'] = 128 # num channels for second last layer output. 
params['z_dim'] = 512
params['disc_iter'] = 1
params['gen_iter'] = 1
params['enc_iter'] = 1
params['image_prepro'] = 'DCGAN' # (/250.; -0.5; /0.5) taken from DCGAN repo.
params['loss_comments'] = 'original setup, but with reversed KL'
params['lambda_adv'] = 1
params['lambda_recon'] = 0
params['lambda_hidden'] = 1e-6
params['epoch_iter'] = 50
params['test'] = True

generator_layers = generator(num_units=params['num_gen_units'], z_dim=params['z_dim'])
discriminator_layers = discriminator()
encoder_layers = encoder(z_dim=params['z_dim'])
generator = generator_layers[-1]
critic = discriminator_layers[-1]
encoder = encoder_layers[-1]

dh = DataHandler(time_len=8, skip_frames=1, num_batches=params['epoch_iter']+1)
eh = ExpHandler(params, test=params['test'])

# placeholders 
images = T.tensor4('images from dataset')
index = T.lscalar() # index to a [mini]batch
eta = theano.shared(lasagne.utils.floatX(params['initial_eta']))
a, b, c = 0, 1, 1

# params
gen_params = ll.get_all_params(generator, trainable=True)
critic_params = ll.get_all_params(critic, trainable=True)
enc_params = ll.get_all_params(encoder, trainable=True)

# outputs 
enc_output = ll.get_output(encoder, inputs=images)
gen_enc_output = ll.get_output(generator, inputs=enc_output)
enc_output_test = ll.get_output(encoder, inputs=images, deterministic=True) 
gen_enc_output_test = ll.get_output(generator, inputs=enc_output_test, deterministic=True)
# gen_noise_output = ll.get_output(generator)

real_out = ll.get_output(critic, inputs=images)
fake_out = ll.get_output(critic, inputs=gen_enc_output)
# fake_noise_out = ll.get_output(critic, inputs=gen_noise_output)

# hidden layers for "feature matching"
hid_real = ll.get_output(discriminator_layers[-3], inputs=images, deterministic=False)
hid_fake = ll.get_output(discriminator_layers[-3], inputs=gen_enc_output, deterministic=False)
 
# losses 
critic_loss = (lasagne.objectives.squared_error(real_out, b).mean() +
 		 lasagne.objectives.squared_error(fake_out, a).mean())         
adv_loss = lasagne.objectives.squared_error(fake_out, c).mean()
hidden_loss = lasagne.objectives.squared_error(hid_fake, hid_real).mean()
recon_loss = lasagne.objectives.squared_error(gen_enc_output, images).mean()

enc_mu, enc_logsigma, l_z = ll.get_output(encoder_layers[-3:], inputs=images)
kl_div = 0.5 * T.sum(1 + enc_logsigma - enc_mu**2 - T.exp(log_sigma), axis=1)
gen_loss = params['lambda_recon'] * recon_loss + params['lambda_adv'] * adv_loss + params['lambda_hidden'] * hidden_loss
enc_loss = hidden_loss + kl_div.mean()

gen_grads = theano.grad(gen_loss, wrt=gen_params)
critic_grads = theano.grad(critic_loss, wrt=critic_params)
enc_grads = theano.grad(enc_loss, wrt=enc_params)
gen_grads_norm = sum(T.sum(T.square(grad)) for grad in gen_grads) / len(gen_grads)
critic_grads_norm = sum(T.sum(T.square(grad)) for grad in critic_grads) / len(critic_grads)
enc_grads_norm = sum(T.sum(T.square(grad)) for grad in enc_grads) / len(enc_grads)

# updates
gen_updates= optimizer_factory(
    params['optimizer'], gen_grads, gen_params, params['initial_eta'])
critic_updates = optimizer_factory(
    params['optimizer'], critic_grads, critic_params, params['critic_eta'])
enc_updates = optimizer_factory(
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
gen_fn_output['adv_loss'] = adv_loss
gen_fn_output['recon_loss'] = recon_loss
gen_fn_output['grads_norm'] = gen_grads_norm

enc_fn_output = OrderedDict()
enc_fn_output['hidden_loss'] = hidden_loss
enc_fn_output['fake_out_mean'] = (fake_out).mean()
enc_fn_output['grads_norm'] = enc_grads_norm

eh.add_model('gen', generator_layers,  gen_fn_output)
eh.add_model('disc', discriminator_layers, critic_fn_output)
eh.add_model('enc', encoder_layers, enc_fn_output)

# functions

train_critic = theano.function(inputs=[index], 
                               outputs=critic_fn_output.values(), 
                               updates=critic_updates,
                               givens={images: dh.GPU_image[index * params['batch_size']: 
                                                         (index+1) * params['batch_size']]},
                               name='train_critic')

train_gen =    theano.function(inputs=[index],
                               outputs=gen_fn_output.values(),
                               updates = gen_updates,
                               givens={images: dh.GPU_image[index * params['batch_size']: 
                                                         (index+1) * params['batch_size']]},
                               name='train_gen') 

train_enc =    theano.function(inputs=[index],
                               outputs=enc_fn_output.values(),
                               updates = enc_updates,
                               givens={images: dh.GPU_image[index * params['batch_size']: 
                                                         (index+1) * params['batch_size']]},
                            name='train_enc')

test_gen =     theano.function(inputs=[index],
                               outputs=[gen_enc_output_test],
                               givens={images: dh.GPU_image[index * params['batch_size']: 
                                                         (index+1) * params['batch_size']]}, 
                               name='test_gen')

'''
training section 
'''
print 'staring training'
for epoch in range(3000000):
    gen_err = 0
    disc_err = 0
    enc_err = 0

    for _ in range(params['epoch_iter']):
        batch_no = dh.get_next_batch_no()
        for _ in range (params['gen_iter']) : 
            gen_out = np.array(train_gen(batch_no))
            gen_err += gen_out
            eh.record('gen', gen_out)
        for _ in range (params['disc_iter']) :
            disc_out = np.array(train_critic(batch_no))
            disc_err += disc_out
            eh.record('disc', disc_out)
        for _ in range (params['enc_iter']) :
            enc_out = np.array(train_enc(batch_no))
            enc_err += enc_out
            eh.record('enc', enc_out)

    # test out model
    batch_no = dh.get_next_batch_no()
    eh.save_image(test_gen(batch_no)[0]); 
    eh.save_image(dh.GPU_image.get_value()[batch_no * params['batch_size']: 
                                       (batch_no+1) * params['batch_size']], real_img=True)
    eh.end_of_epoch()
    
    print epoch

    if params['gen_iter'] > 0 : 
        print("gen  loss:\t\t{}".format(gen_err / ( params['epoch_iter'] * params['gen_iter'])))
    if params['disc_iter'] > 0 : 
        print("disc loss:\t\t{}".format(disc_err / ( params['epoch_iter'] * params['disc_iter'])))
    if params['enc_iter'] > 0 : 
        print("enc loss:\t\t{}".format(enc_err / ( params['epoch_iter'] * params['enc_iter'])))
    
 
