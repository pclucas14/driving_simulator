from model import * 

# hyperparameters
batch_size = 64
initial_eta = 1e-4
load_params = True

generator_layers = generator(num_units=128)
discriminator_layers = discriminator()
generator = generator_layers[-1]
critic = discriminator_layers[-1]

if load_params : 
    load_model(generator, 'gen', 1900)
    load_model(critic, 'disc', 1900)
    print 'params loaded'

# placeholders 
images = T.tensor4('images from dataset')
eta = theano.shared(lasagne.utils.floatX(initial_eta))
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
gen_updates= lasagne.updates.adam(
    gen_grads, gen_params, learning_rate=initial_eta)
critic_updates = lasagne.updates.adam(
    critic_grads, critic_params, learning_rate=initial_eta)

# functions 
train_critic = theano.function(inputs=[images], 
                               outputs=[(real_out).mean(),
                                        (fake_out).mean(),
                                        critic_loss,
                                        critic_grads_norm], 
                               updates=critic_updates,
                               name='train_critic') #, on_unused_input='warn')

train_gen =    theano.function(inputs=[],
                               outputs=[(fake_out > 0.5).mean(), 
                                        (fake_out).mean(),
                                        gen_loss,
                                        gen_grads_norm], 
                               updates = gen_updates,
                               name='train_gen') #, on_unused_input='warn')

test_gen =     theano.function(inputs=[],
                               outputs=[gen_output], 
                               name='test_gen')#, on_unused_input='warn')

'''
training section 
'''
print 'loading dataset'
dataset = load_dataset(sample=False)
# import pdb; pdb.set_trace()
num_batches = dataset.shape[0] / batch_size - 2
batches = iterate_minibatches(dataset[:num_batches * batch_size], batch_size, shuffle=True, forever=True)

print 'staring training'
for epoch in range(3000000):
    gen_err = 0
    disc_err = 0

    for _ in range(50):
        target = next(batches)
        for _ in range (1) : 
	    gen_err += np.array(train_gen())
        disc_err += np.array(train_critic(target))

    # test out
    samples = test_gen()[0]
    samples *= 0.5; samples += 0.5; samples *= 255.
    print epoch
    print("gen  loss:\t\t{}".format(gen_err / 50. ))
    print("disc loss:\t\t{}".format(disc_err / 50. ))
    saveImage(samples, epoch)

    if epoch % 25 == 0 : 
        save_model(critic, 'disc', epoch)
        save_model(generator, 'gen', epoch)
