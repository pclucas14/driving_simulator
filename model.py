import theano
import theano.tensor as T
import lasagne
import lasagne.layers as  ll
import pdb
import numpy as np
import numpy.random as rng
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams
from utils import * 
from layers import * 


def encoder(z_dim=100, input_var=None, num_units=512, vae=True):
    encoder = []
    lrelu = lasagne.nonlinearities.LeakyRectify(0.2)

    encoder.append(ll.InputLayer(shape=(None, 3, 80, 160), input_var=input_var))

    encoder.append(ll.Conv2DLayer(encoder[-1], 
                                                      num_filters=num_units/8,
                                                      filter_size=(5,5),
                                                      stride=2, 
                                                      pad=2, 
                                                      nonlinearity=lrelu))

    encoder.append(ll.batch_norm(ll.Conv2DLayer(encoder[-1], 
                                                      num_filters=num_units/4,
                                                      filter_size=(5,5),
                                                      stride=2, 
                                                      pad=2, 
                                                      nonlinearity=lrelu)))

    encoder.append(ll.batch_norm(ll.Conv2DLayer(encoder[-1], 
                                                      num_filters=num_units/2,
                                                      filter_size=(5,5),
                                                      stride=2, 
                                                      pad=2, 
                                                      nonlinearity=lrelu)))

    encoder.append(ll.batch_norm(ll.Conv2DLayer(encoder[-1], 
                                                      num_filters=num_units,
                                                      filter_size=(5,5),
                                                      stride=2, 
                                                      pad=2, 
                                                      nonlinearity=lrelu)))

    encoder.append(ll.FlattenLayer(encoder[-1]))

    if vae : 
        enc_mu = ll.DenseLayer(encoder[-1], num_units=z_dim, nonlinearity=None)

        enc_logsigma = ll.DenseLayer(encoder[-1], num_units=z_dim, nonlinearity=None)

        l_z = GaussianSampleLayer(enc_mu, enc_logsigma, name='Z layer')

        encoder += [enc_mu, enc_logsigma, l_z]
        

    for layer in encoder : 
        print layer.output_shape
    print ""

    return encoder




# generates tensors of shape (None, 3, 80, 160)
def generator(z_dim=100, num_units=128, input_var=None, batch_size=64):
    generator = []

    theano_rng = RandomStreams(rng.randint(2 ** 15))
    noise = theano_rng.normal(size=(batch_size, z_dim), avg=0.0, std=1.0)
    input_var = noise if input_var is None else input_var

    generator.append(ll.InputLayer(shape=(batch_size, z_dim), input_var=input_var))

    generator.append(ll.DenseLayer(generator[-1], num_units*8*5*10))

    generator.append(ll.ReshapeLayer(generator[-1], shape=(-1, num_units*8, 5, 10)))

    generator.append(ll.batch_norm(ll.TransposedConv2DLayer(generator[-1],
                                                            num_filters=num_units*4, 
                                                            filter_size=(4,4), 
                                                            stride=(2,2), 
                                                            crop=1)))

    generator.append(ll.batch_norm(ll.TransposedConv2DLayer(generator[-1],
                                                            num_filters=num_units*2, 
                                                            filter_size=(4,4), 
                                                            stride=(2,2), 
                                                            crop=1)))

    generator.append(ll.batch_norm(ll.TransposedConv2DLayer(generator[-1],
                                                            num_filters=num_units, 
                                                            filter_size=(4,4), 
                                                            stride=(2,2), 
                                                            crop=1)))

    generator.append(ll.TransposedConv2DLayer(generator[-1],
                                                            num_filters=3, 
                                                            filter_size=(4,4), 
                                                            stride=(2,2), 
                                                            crop=1, 
                                                            nonlinearity=T.tanh))

    for layer in generator : 
        print layer.output_shape
    print ""

    return generator


# takes images of shape (None, 3, 80, 160) and returns score (LSGAN setup)
def discriminator(input_var=None, num_units=512):
    discriminator = []
    lrelu = lasagne.nonlinearities.LeakyRectify(0.2)

    discriminator.append(ll.InputLayer(shape=(None, 3, 80, 160), input_var=input_var))

    discriminator.append(ll.Conv2DLayer(discriminator[-1], 
                                                      num_filters=num_units/8,
                                                      filter_size=(5,5),
                                                      stride=2, 
                                                      pad=2, 
                                                      nonlinearity=lrelu))

    discriminator.append(ll.batch_norm(ll.Conv2DLayer(discriminator[-1], 
                                                      num_filters=num_units/4,
                                                      filter_size=(5,5),
                                                      stride=2, 
                                                      pad=2, 
                                                      nonlinearity=lrelu)))
                                                    
    discriminator.append(ll.batch_norm(ll.Conv2DLayer(discriminator[-1], 
                                                      num_filters=num_units/2,
                                                      filter_size=(5,5),
                                                      stride=2, 
                                                      pad=2, 
                                                      nonlinearity=lrelu)))

    discriminator.append(ll.batch_norm(ll.Conv2DLayer(discriminator[-1], 
                                                      num_filters=num_units,
                                                      filter_size=(5,5),
                                                      stride=2, 
                                                      pad=2, 
                                                      nonlinearity=lrelu)))

    discriminator.append(ll.FlattenLayer(discriminator[-1]))

    discriminator.append(ll.DenseLayer(discriminator[-1], num_units=1,nonlinearity=None))


    for layer in discriminator : 
        print layer.output_shape
    print ""

    return discriminator


# takes images of shape (None, seq_length, 3, 80, 160) and returns score (LSGAN setup)
def discriminator_3D(input_var=None, num_units=512, seq_length=4):
    discriminator = []
    lrelu = lasagne.nonlinearities.LeakyRectify(0.2)

    discriminator.append(ll.InputLayer(shape=(None, seq_length, 3, 80, 160), input_var=input_var))
    
    # lasagne documentations requires shape :
    # (batch_size, num_input_channels, input_depth, input_rows, input_columns)
    # so we need to change dimension ordering
    
    discriminator.append(ll.DimshuffleLayer(discriminator[-1], (0, 2, 1, 3, 4)))

    discriminator.append(ll.Conv3DLayer(discriminator[-1], 
                                                      num_filters=num_units/8,
                                                      filter_size=5,
                                                      stride=2, 
                                                      pad=2, 
                                                      nonlinearity=lrelu))

    discriminator.append(ll.batch_norm(ll.Conv3DLayer(discriminator[-1], 
                                                      num_filters=num_units/4,
                                                      filter_size=5,
                                                      stride=2, 
                                                      pad=2, 
                                                      nonlinearity=lrelu)))
                                                    
    discriminator.append(ll.batch_norm(ll.Conv3DLayer(discriminator[-1], 
                                                      num_filters=num_units/2,
                                                      filter_size=5,
                                                      stride=2, 
                                                      pad=2, 
                                                      nonlinearity=lrelu)))

    discriminator.append(ll.batch_norm(ll.Conv3DLayer(discriminator[-1], 
                                                      num_filters=num_units,
                                                      filter_size=5,
                                                      stride=2, 
                                                      pad=2, 
                                                      nonlinearity=lrelu)))
    
    discriminator.append(ll.FlattenLayer(discriminator[-1]))

    discriminator.append(ll.DenseLayer(discriminator[-1], num_units=1,nonlinearity=None))


    for layer in discriminator : 
        print layer.output_shape
    print ""

    return discriminator


# recurrent structure for latent space prediction. 
def recurrent(input_var=None, num_units=512, batch_size=64, grad_clip=100):
    recurrent = []

    theano_rng = RandomStreams(rng.randint(2 ** 15))
    # we want noise to match tanh range of activation ([-1,1])  
    noise = theano_rng.uniform(size=(batch_size, 1, num_units), low=-1.0, high=1.0)
    input_var = noise if input_var is None else input_var
    
    recurrent.append(ll.InputLayer(shape=(batch_size, 1, num_units), input_var=input_var))

    recurrent.append(ll.LSTMLayer(recurrent[-1], 
                                  num_units, 
                                  grad_clipping=grad_clip)) #tanh is default

    

    for layer in recurrent : 
        print layer.output_shape
    print ""

    return recurrent

    




