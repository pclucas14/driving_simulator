from model import * 

# hyperparameters
batch_size = 64
initial_eta = 1e-4
load_params = False

generator_layers = generator(num_units=128)
discriminator_layers = discriminator(num_units=512)
encoder_layers = encoder(num_units=512)

generator = generator_layers[-1]
critic = discriminator_layers[-1]
encoder = encoder_layers[-1]

if load_params : 
    last_saved_epoch = 1900
    load_model(generator, 'gen', last_saved_epoch)
    load_model(critic, 'disc', last_saved_epoch)
    load_model(encoder, 'enc', last_saved_epoch)
    print 'params loaded'

# placeholders 
images = T.tensor4('images from dataset')
eta = theano.shared(lasagne.utils.floatX(initial_eta))
a, b, c = 0, 1, 1
print 'initializing functions'

# params
gen_params = ll.get_all_params(generator, trainable=True)
critic_params = ll.get_all_params(critic, trainable=True)
enc_params = ll.get_all_params(encoder, trainable=True)

# outputs
enc_output = ll.get_output(encoder, inputs=images)
gen_enc_output = ll.get_output(generator, inputs=enc_output)
gen_noise_output = ll.get_output(generator)

real_out = ll.get_output(critic, inputs=images)
fake_enc_out = ll.get_output(critic, inputs=gen_enc_output)
fake_noise_out = ll.get_output(critic, inputs=gen_noise_output)

# hidden layers for "feature matching"
hid_real = ll.get_output(discriminator_layers[-3], inputs=images, deterministic=False)
hid_fake = ll.get_output(discriminator_layers[-3], inputs=gen_enc_output, deterministic=False)
 
# losses 
enc_mu, enc_logsigma, l_z = ll.get_output(encoder_layers[-3:], inputs=images)
kl_div = -0.5 * T.sum(1 + 2*enc_logsigma - T.sqr(enc_mu) - T.exp(2 * enc_logsigma))
like_loss = lasagne.objectives.squared_error(hid_fake, hid_real).mean()
recon_vs_gan = 1e-6

critic_loss = (lasagne.objectives.squared_error(real_out, b).mean() +
 		       lasagne.objectives.squared_error(fake_enc_out, a).mean() + 
               lasagne.objectives.squared_error(fake_noise_out, a).mean()) 
        
gen_loss = (lasagne.objectives.squared_error(fake_enc_out, c).mean() + 
            lasagne.objectives.squared_error(fake_noise_out, c).mean())
gen_loss += recon_vs_gan * like_loss

enc_loss = kl_div + like_loss

gen_grads = theano.grad(gen_loss, wrt=gen_params)
critic_grads = theano.grad(critic_loss, wrt=critic_params)
enc_grads = theano.grad(enc_loss, wrt=enc_params)
gen_grads_norm = sum(T.sum(T.square(grad)) for grad in gen_grads) / len(gen_grads)
critic_grads_norm = sum(T.sum(T.square(grad)) for grad in critic_grads) / len(critic_grads)
enc_grads_norm = sum(T.sum(T.square(grad)) for grad in enc_grads) / len(enc_grads)

# updates
gen_updates= lasagne.updates.adam(
    gen_grads, gen_params, learning_rate=initial_eta)
critic_updates = lasagne.updates.adam(
    critic_grads, critic_params, learning_rate=initial_eta)
enc_updates = lasagne.updates.adam(
    enc_grads, enc_params, learning_rate=initial_eta)

     
# functions 
train_critic =   theano.function(inputs=[images], 
                                 outputs=[(real_out).mean(),
                                        fake_enc_out.mean() + fake_noise_out.mean(),
                                        critic_loss,
                                        critic_grads_norm], 
                                 updates=critic_updates,
                                 name='train_critic') #, on_unused_input='warn')

train_gen =      theano.function(inputs=[images],
                                 outputs=[(fake_enc_out).mean(), 
                                        (fake_noise_out).mean(),
                                        like_loss,
                                        gen_loss,
                                        gen_grads_norm], 
                                 updates = gen_updates,
                                 name='train_generator') #, on_unused_input='warn')

train_enc =      theano.function(inputs=[images],
                                 outputs=[like_loss, 
                                        kl_div,
                                        enc_loss,
                                        enc_grads_norm], 
                                 updates = enc_updates,
                                 name='train_encoder') #, on_unused_input='warn')

test_gen_noise = theano.function(inputs=[],
                                 outputs=[gen_noise_output], 
                                 name='test_gen')#, on_unused_input='warn')

test_gen_enc =  theano.function(inputs=[images],
                                outputs=[gen_enc_output], 
                                name='test_gen')#, on_unused_input='warn')

'''
training section 
'''
print 'loading dataset'
dataset = load_dataset(sample=False)
num_batches = dataset.shape[0] / batch_size - 2
batches = iterate_minibatches(dataset[:num_batches * batch_size], batch_size, shuffle=True, forever=True)
test_batch = dataset[num_batches * batch_size : (num_batches + 1) * batch_size]
test_batch_copy = (test_batch * 0.5 + 0.5) * 255. 

print 'staring training'
for epoch in range(3000000):
    gen_err = 0
    disc_err = 0
    enc_err = 0

    for _ in range(50):
        target = next(batches)
        gen_err += np.array(train_gen(target))
        disc_err += np.array(train_critic(target))
        enc_err += np.array(train_enc(target))

    # test out
    samples = test_gen_enc(test_batch)[0]
    samples *= 0.5; samples += 0.5; samples *= 255.
    saveImage(samples, epoch)
    saveImage(test_batch_copy, 0, name='og_')

    print epoch
    print("gen  loss:\t\t{}".format(gen_err / 50. ))
    print("disc loss:\t\t{}".format(disc_err / 50. ))
    print("enc  loss:\t\t{}".format(enc_err / 50. ))

    if epoch % 25 == 0 : 
        save_model(critic, 'disc', epoch)
        save_model(generator, 'gen', epoch)
        save_model(encoder, 'enc', epoch)