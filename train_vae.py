from model import * 

# hyperparameters
batch_size = 64
initial_eta = 2e-4

generator_layers = generator()
discriminator_layers = discriminator()
generator = generator_layers[-1]
critic = discriminator_layers[-1]

# placeholders 
images = T.tensor4('images from dataset')
eta = theano.shared(lasagne.utils.floatX(initial_eta))
a, b, c = 0, 1, 1

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
                               outputs=[(fake_out).mean(), 
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
dataset = load_dataset_dummy()#(sample=False)
num_batches = dataset.shape[0] / batch_size - 2
batches = iterate_minibatches(dataset[:num_batches * batch_size], batch_size, shuffle=True, forever=True)

for epoch in range(3000000):
    gen_err = 0
    disc_err = 0

    for _ in range(0):
        target = next(batches)
        gen_err += np.array(train_gen())
        disc_err += np.array(train_critic(target))

    # test out
    samples = test_gen()[0]
    # import pdb; pdb.set_trace()
    samples *= 0.5
    samples += 0.5
    samples *= 255.
    print epoch
    # print 'generator stats', gen_err / 50.
    # print 'discriminator stats', disc_err / 50.

    saveImage(samples, epoch)