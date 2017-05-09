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


def encoder(z_dim=100, input_var=None, num_units=512):
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
    
    #noise = theano_rng.uniform(size=(batch_size, z_dim)) 
    noise = theano_rng.normal(size=(batch_size, z_dim), avg=0.0, std=1.0)

    generator.append(ll.InputLayer(shape=(batch_size, z_dim), input_var=noise))

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





