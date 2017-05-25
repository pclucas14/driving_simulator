import theano
import theano.tensor as T
import lasagne
import lasagne.layers as  ll
import pdb
import numpy as np
import numpy.random as rng
import time
import threading
from utils import * 
from layers import *

'''
This codes allows the user to preprocess the next set of minibatches
while the model is training. This eliminates most of the overhead of 
1) copying to GPU 2) preprocessing the data.
'''

def preprocess_next_batch(dataHandler):
    dataHandler.next_batch_ready = False
    # print 'started preprocessing next batch'
    # note that dataHandler's iterator returns a preprocessed 
    # dataset, it's just a really slow process, hence the thread.
    other_features, images = next(dataHandler.iterator)
    # TODO: remove this
    # images = extract_video_frames(images)

    if other_features.shape != dataHandler.other_shape : 
        other_features = other_features.reshape(dataHandler.other_shape)
    
    if images.shape != dataHandler.image_shape : 
        images = images.reshape(dataHandler.image_shape)

    if images.shape[-1] == 3 : 
        # tensorflow ordering --> theano ordering
        print 'reordered image tensor'
        images = images.transpose(0,3,1,2)
    
    dataHandler.next_batch_image = images
    dataHandler.next_batch_other = other_features
    # print 'done processing next batch!'
    dataHandler.next_batch_ready = True
    
        

# responsible for efficiently delivering minibatches 
class DataHandler():
    def __init__(self, batch_size=64, num_batches=51, num_extra_features=2, time_len=1, skip_frames=1):
        assert num_batches * batch_size % skip_frames == 0 
        # TODO : remove the extra True condition
        if time_len == 1 : #or True : 
            self.GPU_image = theano.shared(np.zeros((num_batches * batch_size / skip_frames * time_len, 3, 80, 160)).astype('float32'))
            self.GPU_other = theano.shared(np.zeros((num_batches * batch_size / skip_frames * time_len, num_extra_features)).astype('float32'))

            self.next_batch_image = np.zeros((num_batches * batch_size / skip_frames * time_len, 3, 80, 160)).astype('float32')
            self.next_batch_other = np.zeros((num_batches * batch_size / skip_frames * time_len, num_extra_features)).astype('float32')

            self.image_shape = (num_batches * batch_size / skip_frames * time_len, 3, 80, 160)
            self.other_shape = (num_batches * batch_size / skip_frames * time_len, num_extra_features)
        else : 
            self.GPU_image = theano.shared(np.zeros((num_batches * batch_size, time_len / skip_frames, 3, 80, 160)).astype('float32'))
            self.GPU_other = theano.shared(np.zeros((num_batches * batch_size, time_len / skip_frames, num_extra_features)).astype('float32'))
        
            self.next_batch_image = np.zeros((num_batches * batch_size, time_len / skip_frames, 3, 80, 160)).astype('float32')
            self.next_batch_other = np.zeros((num_batches * batch_size, time_len / skip_frames, num_extra_features)).astype('float32')      

            self.image_shape = (num_batches * batch_size, time_len / skip_frames, 3, 80, 160)
            self.other_shape = (num_batches * batch_size, time_len / skip_frames, num_extra_features)  
    
        self.num_batches = num_batches
        self.batch_size = batch_size
        self.num_extra_features = num_extra_features
        self.next_batch_ready = False
        # skip_frame = 1 means no skipping frames, = 2 means skip 1 out of 2 and so on
        self.skip_frames = skip_frames
    
        self.preprocessing_thread = self.reset_thread()
        # TODO : fix next line
        self.iterator = data_iterator(num_batches*batch_size, time_len=time_len, skip_frames=skip_frames)

        # load the first minibatch
        preprocess_next_batch(self)
        self.current_batch = self.num_batches
        # ^^ ensures that the 1st iteration loads images on GPU



    # returns the index of the next batch. Takes care of swapping in a new batch
    # on GPU if necessary
    def get_next_batch_no(self):
        if self.current_batch < self.num_batches - 1: 
            self.current_batch += 1
            return self.current_batch
        else : 
            waited = 0
            while not self.next_batch_ready : 
                waited += 1
                time.sleep(5)
            # print("waited {} seconds for next batch".format(str(waited*5)))
        
            # reset the thread for the next iteration
            self.preprocessing_thread = self.reset_thread()
       
            # load ready batch on GPU
            self.update_GPU_batch()
    
            # prepare the next batch
            self.preprocessing_thread.start()
    
            self.current_batch = 0
            return self.current_batch



    # takes the ready batch and puts it on GPU.
    def update_GPU_batch(self):
        assert self.next_batch_ready == True
        self.GPU_image.set_value(self.next_batch_image.astype('float32'))
        self.GPU_other.set_value(self.next_batch_other.astype('float32'))



    # creates a new instance of the preprocessing thread.
    def reset_thread(self):
        return threading.Thread(target=preprocess_next_batch, args=(self,))




    
    

    

