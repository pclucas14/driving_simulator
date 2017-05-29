from model import * 
from DataHandler import * 
from ExpHandler import * 
from collections import OrderedDict 

# hyperparameters / parameters
params = OrderedDict()
params['batch_size'] = 64
params['initial_eta'] = 1e-4
params['critic_eta'] = 2e-3
params['load_weights'] = None#(140, 1900) # version/epoch tupple pair None
params['optimizer'] = 'rmsprop'
params['num_gen_units'] = 128 # num channels for second last layer output. 
params['z_dim'] = 512
params['disc_iter'] = 1
params['gen_iter'] = 1
params['enc_iter'] = 1
params['rec_iter'] = 1
params['epoch_iter'] = 25
params['seq_length'] = 128
params['skip_frames'] = 4
params['lstm_seq_length'] = 4
params['disc_seq_length'] = 16
params['lambda_adv'] = 0.5
params['lambda_recon'] = 50
params['comments'] = '1st try with ae_rgan v3'
params['real_frame_in_fake_input_amt'] = 1
params['image_prepro'] = 'DCGAN' # ( /250.; -0.5; /0.5) taken from DCGAN repo.
a, b, c = 0, 1, 1

# lasagne models
generator_layers = generator(num_units=params['num_gen_units'], z_dim=params['z_dim'])
discriminator_layers = discriminator_3D(seq_length=params['disc_seq_length'])
encoder_layers = encoder(vae=False, z_dim=params['z_dim'])
recurrent_layers = recurrent(seq_length=params['lstm_seq_length'])
generator = generator_layers[-1]
critic = discriminator_layers[-1]
encoder = encoder_layers[-1]
recurrent = recurrent_layers[-1]
reducer, expander = encoder_adapter(seq_length=params['seq_length']/params['skip_frames'])

# helper classes
dh = DataHandler(time_len=params['seq_length'], 
                 skip_frames=params['skip_frames'], 
                 num_batches=params['epoch_iter']+1)
eh = ExpHandler(params, dum_freq=500)

# placeholders 
images = T.tensor5('images from dataset')
index = T.lscalar() # index to a [mini]batch
eta = theano.shared(lasagne.utils.floatX(params['initial_eta']))

# intermediate tensors 
four_d_images = ll.get_output(reducer, inputs=images)
two_d_enc_output = ll.get_output(encoder, inputs=four_d_images)
enc_output = ll.get_output(expander, inputs=two_d_enc_output) # (b_s, seq_length, z_dim)

# recurrent network latent vector prediction
lstm_input = enc_output[:, 0:params['lstm_seq_length'], :] # (b_s, lstm_seq_length, z_dim)
lstm_preds = []
# TODO : this should be for all remaining outputs
for i in range(params['seq_length']/params['skip_frames']-params['lstm_seq_length']):
    lstm_preds.append(ll.get_output(recurrent, inputs=lstm_input))
    # update lstm_input with the latest prediction
    lstm_input = T.concatenate([lstm_input, lstm_preds[-1]], axis=1)
    # remove 1st input
    lstm_input = lstm_input[:, slice(1, 1 + params['lstm_seq_length']), :]

# pushing latent vector through generator to get a sequence of frames
gen_outputs_hallucinate = []
gen_outputs_recon = []
for i in range(params['disc_seq_length']):
    gen_input = enc_output[:, slice(i,i+1), :]
    gen_input = gen_input.reshape((gen_input.shape[0], gen_input.shape[2]))
    gen_outputs_recon.append(ll.get_output(generator, inputs=gen_input))
for i in range(len(lstm_preds)):
    gen_input = lstm_preds[i]
    gen_input = gen_input.reshape((gen_input.shape[0], gen_input.shape[2]))
    gen_outputs_hallucinate.append(ll.get_output(generator, inputs=gen_input))

critic_input_hallucinate = T.stack(gen_outputs_hallucinate, axis=1)
critic_input_recon = T.stack(gen_outputs_recon, axis=1)
lstm_preds_ = T.concatenate(lstm_preds, axis=1)

if params['real_frame_in_fake_input_amt'] > 0 or len(lstm_preds) != params['disc_seq_length'] : 
    index_s = params['seq_length']/params['skip_frames'] - params['disc_seq_length']
    # for debugging purposes
    assert index_s == 3
    first_frame = images[:, slice(index_s, index_s+1), :, :, :]
    # drop the first predicted frame
    # critic_input_hallucinate = critic_input_hallucinate[:, 1:, :, :, :]
    # add the first real frame
    critic_input_hallucinate = T.concatenate([first_frame, critic_input_hallucinate], axis=1)

# params
gen_params = ll.get_all_params(generator, trainable=True)
critic_params = ll.get_all_params(critic, trainable=True)
enc_params =   ll.get_all_params(encoder, trainable=True)
rec_params = ll.get_all_params(recurrent, trainable=True)

# discriminator outputs
real_out = ll.get_output(critic, inputs=images[:, slice(0,params['disc_seq_length']), :, :, :])
fake_recon_out = ll.get_output(critic, inputs=critic_input_recon)
fake_hallucinate_out = ll.get_output(critic, inputs=critic_input_hallucinate)

# losses 
critic_loss = (lasagne.objectives.squared_error(real_out, b).mean() +
               lasagne.objectives.squared_error(fake_recon_out, a).mean() +
               lasagne.objectives.squared_error(fake_hallucinate_out, a).mean())
recon_loss = lasagne.objectives.squared_error(critic_input_recon, images[:, 0:params['disc_seq_length'], :, :, :]).mean()
adv_recon_loss = lasagne.objectives.squared_error(fake_recon_out, c).mean()
adv_hallucinate_loss = lasagne.objectives.squared_error(fake_hallucinate_out, c).mean()
mse_enc_loss = lasagne.objectives.squared_error(lstm_preds_, enc_output[:, -lstm_preds_.shape[1]:, :]).mean()
gen_loss = params['lambda_recon'] * recon_loss + params['lambda_adv'] * (adv_recon_loss + adv_hallucinate_loss)
rec_loss = params['lambda_recon'] * mse_enc_loss + params['lambda_adv'] * adv_hallucinate_loss
enc_loss = params['lambda_recon'] * recon_loss + params['lambda_adv'] * (adv_recon_loss)


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
    params['optimizer'], enc_grads, enc_params, params['initial_eta'])
rec_updates = optimizer_factory(
    params['optimizer'], rec_loss, rec_params, params['initial_eta'])
gen_rec_updates = gen_updates.copy()
gen_rec_updates.update(rec_updates)

# function outputs
critic_fn_output = OrderedDict()
critic_fn_output['real_out_mean'] = (real_out).mean()
critic_fn_output['fake_recon_out_mean'] = (fake_recon_out).mean()
critic_fn_output['fake_hallucinate_out_mean'] = (fake_hallucinate_out).mean()
critic_fn_output['loss'] = critic_loss
critic_fn_output['grads_norm'] = critic_grads_norm

enc_fn_output = OrderedDict()
enc_fn_output['recon_loss'] = recon_loss
enc_fn_output['grads_norm'] = enc_grads_norm
enc_fn_output['fake_recon_out_mean'] = (fake_recon_out).mean()

gen_fn_output = OrderedDict()
gen_fn_output['fake_recon_out_mean'] = (fake_recon_out).mean()
gen_fn_output['fake_hallucinate_out_mean'] = (fake_hallucinate_out).mean()
gen_fn_output['recon_loss'] = recon_loss
gen_fn_output['rec_mse_loss'] = mse_enc_loss
gen_fn_output['grads_norm'] = gen_grads_norm

rec_fn_output = OrderedDict()
rec_fn_output['fake_hallucinate_out_mean'] = (fake_hallucinate_out).mean()
rec_fn_output['mse_rec_loss'] = mse_enc_loss

eh.add_model('gen', generator_layers,  gen_fn_output)
eh.add_model('disc', discriminator_layers, critic_fn_output)
eh.add_model('enc', encoder_layers, enc_fn_output)
eh.add_model('rec', recurrent_layers, rec_fn_output)

# functions
print 'compiling functions'
'''
test_shapes = theano.function(inputs=[index],
                              outputs=[critic_input_hallucinates],#, critic_input_recon],
                             givens={images: dh.GPU_image[index * params['batch_size']: 
                                                         (index+1) * params['batch_size']]})
'''
train_critic = theano.function(inputs=[index], 
                               outputs=critic_fn_output.values(), 
                               updates=critic_updates,
                               givens={images: dh.GPU_image[index * params['batch_size']: 
                                                         (index+1) * params['batch_size']]},
                               name='train_critic')
print 'compiled critic function'

train_gen =    theano.function(inputs=[index],
                               outputs=gen_fn_output.values(),
                               updates = gen_rec_updates,
                               givens={images: dh.GPU_image[index * params['batch_size']: 
                                                         (index+1) * params['batch_size']]},
                               name='train_gen', on_unused_input='warn') 
print 'compiled gen function'

train_enc =    theano.function(inputs=[index],
                               outputs=enc_fn_output.values(),
                               updates = enc_updates,
                               givens={images: dh.GPU_image[index * params['batch_size']: 
                                                         (index+1) * params['batch_size']]},
                               name='train_enc')
print 'compiled enc function'

train_rec =    theano.function(inputs=[index],
                               outputs=rec_fn_output.values(),
                               updates = rec_updates,
                               givens={images: dh.GPU_image[index * params['batch_size']: 
                                                         (index+1) * params['batch_size']]},
                               name='train_enc')

test_gen =     theano.function(inputs=[index],
                               outputs=[critic_input_recon, critic_input_hallucinate],
                               givens={images: dh.GPU_image[index * params['batch_size']: 
                                                         (index+1) * params['batch_size']]}, 
                               name='test_gen')
print 'compiled test function'


'''
training section 
'''
print 'staring training'
for epoch in range(3000000):
    gen_err, disc_err, enc_err, rec_err = 0, 0, 0, 0

    for _ in range(params['epoch_iter']):
        batch_no = dh.get_next_batch_no()
        #out = test_shapes(batch_no)
        #import pdb; pdb.set_trace()
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
        for _ in range (params['rec_iter']) :
            rec_out = np.array(train_rec(batch_no))
            rec_err += rec_out
            eh.record('rec', rec_out)

    # test out model
    recon, hallucinate = test_gen(batch_no)
    # import pdb; pdb.set_trace()
    test_imgs = extract_video_frames(hallucinate)
    eh.save_image(test_imgs)
    test_imgs = extract_video_frames(recon)
    eh.save_image(test_imgs, extra='_recon')
    real_imgs = dh.GPU_image.get_value()[batch_no*params['batch_size']:(batch_no+1)*params['batch_size']] 
    real_imgs = extract_video_frames(real_imgs)
    # eh.save_image(test_imgs)
    eh.save_image(real_imgs, real_img=True)
    eh.end_of_epoch()
    
    print epoch

    if params['gen_iter'] > 0 : 
        print("gen  loss:\t\t{}".format(gen_err / ( params['epoch_iter'] * params['gen_iter'])))
    if params['disc_iter'] > 0 : 
        print("disc loss:\t\t{}".format(disc_err / ( params['epoch_iter'] * params['disc_iter'])))
    if params['enc_iter'] > 0 : 
        print("enc loss:\t\t{}".format(enc_err / ( params['epoch_iter'] * params['enc_iter'])))
    if params['rec_iter'] > 0 : 
        print("rec loss:\t\t{}".format(rec_err / ( params['epoch_iter'] * params['rec_iter'])))
