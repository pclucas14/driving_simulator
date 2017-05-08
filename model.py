import theano
import theano.tensor as T
import lasagne
import lasagne.layers as  ll
import pdb
import numpy as np
import numpy.random as rng
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams
from utils import * 


# generates tensors of shape (None, 3, 80, 160)
def generator(z_dim=100, num_units=128, input_var=None, batch_size=64):
    generator = []
    theano_rng = RandomStreams(rng.randint(2 ** 15))
    noise = theano_rng.uniform(size=(batch_size, z_dim)) 

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

    return discriminator




