from model import * 
from collections import OrderedDict 
from layers import * 
from utils import * 
import numpy as np 
import pdb

b_s = 60

f = file('bouncing_mnist_test.npy', 'rb')
data = np.load(f) # (10000, 20, 64, 64)
data = data.reshape((10000, 20, 1, 64, 64))

model = ll.InputLayer(shape=(b_s, 19, 1, 64, 64))
model_lstm = ConvLSTMLayer(model, 120, only_return_final=True)
model = ll.Conv2DLayer(model_lstm, 1, 5, pad='same')

images = T.tensor5('images')
target = T.tensor4('target')

model_out = ll.get_output(model, inputs=images)
loss = lasagne.objectives.squared_error(target, model_out).mean()
params = ll.get_all_params(model, trainable=True)

updates = optimizer_factory('adam', loss, params, 1e-3)

print 'compiling functions'

train_fn =    theano.function(inputs=[images, target],
                               outputs=[loss],
                               updates = updates,
                               name='train_gen') 

test_fn =     theano.function(inputs=[images],
                               outputs=[model_out], 
                               name='test_gen')

print 'starting training'

index = 0
num_batches = data.shape[0] / b_s - 1
for _ in range(1000000):
    for _ in range(50):
        index = (index + 1) % num_batches
        imgs = data[index*b_s:(index+1)*b_s, :-1, :, :, :]
        targ = data[index*b_s:(index+1)*b_s, -1, :, :, :]
        # cpdb.set_trace()
        loss = train_fn(imgs, targ)
    img = test_fn(imgs)
    pdb.set_trace()
        

print model_out.shape

print model_out