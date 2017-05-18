import numpy as np
import h5py
import scipy.misc
from PIL import Image
from model import * 
from os import listdir
from os.path import isfile, join
import time
import lasagne

path = '/home/ml/lpagec/research/dataset/'

'''
method depreciated ?. Should not longer be used (replaced by DataHandler)
'''
def load_dataset(normalize=True, resize=True, sample=False):
    path = '/home/ml/lpagec/research/dataset/'

    # check if processed file in memory : 
    try : 
        f = file(path + 'camera/other/data.bin', "rb")
        print "found cached dataset"
        dataset = np.load(f)
        f.close()
        return dataset
    except : 
        print "no cached version. processing raw data."

	# processing frames
        files_in_cam_dir = [f for f in listdir(path + 'camera/') if isfile(join(path + 'camera/', f))]
        camera_files = [f for f in files_in_cam_dir if '.h5' in f]

        # storing all frames in this dictionnary, key = filename 
        all_frames = dict()
        for camera_path in camera_files : 
            camera_file = h5py.File(path + 'camera/' + camera_path)
            camera_file = camera_file['X']
            camera_file = np.array(camera_file)

            if sample : 
                camera_file = camerafile[:1000]

            if resize :
                # scipy takes tf ordering, so make sure array respects ordering
                if camera_file.shape[-1] != 3 :
                    camera_file = camera_file.transpose(0, 2, 3, 1)

                dataset = np.zeros((camera_file.shape[0], 80, 160, 3))
                for i in range(camera_file.shape[0]):
                    img = scipy.misc.imresize( camera_file[i], (80, 160, 3))
                    dataset[i] = img
                
            else : 
                dataset = camera_file
            '''
            # remove random images at the beginning (first Dataset specific!)
                dataset = dataset[900:]
                dataset = dataset[:-1500]
            '''
            np.random.shuffle(dataset)
            dataset = dataset.astype('float32')
            dataset /= 255.
            
            # put dataset in Theano ordering
            if dataset.shape[1] != 3 : 
                dataset = dataset.transpose(0, 3, 1, 2)

            if normalize : 
                dataset -= 0.5
                dataset /= 0.5

            # save to global dictionnary
            all_frames[camera_path] = dataset
       
        # processing logs
        files_in_log_dir = [f for f in listdir(path + 'log/') if isfile(join(path + 'log/', f))]
        log_files = [f for f in onlyfiles if '.h5' in f]
        
        # storing all log info in this dictionnary, key = filename
        all_logs = dict()
        for log_path in log_files : 
            log_file = h5py.File(path + 'log/' + log_path)
	    all_logs[log_path] = log_file

	# TODO: complete code
        
	# save processed dataset.
        f = file(path + 'data.bin', "wb")
        np.save(f,dataset)
    
        return dataset
        
def load_dataset_dummy():
    return np.ones((1000, 3, 80, 160))


def iterate_minibatches(inputs,  batchsize, full=None, shuffle=False, 
                        forever=False):   
    while True : 
        if shuffle:
            indices = np.arange(len(inputs))
            np.random.shuffle(indices)
        for start_idx in range(0, len(inputs) - batchsize + 1, batchsize):
            if shuffle:
                excerpt = indices[start_idx:start_idx + batchsize]
            else:
                excerpt = slice(start_idx, start_idx + batchsize)
            yield inputs[excerpt]#, targets[excerpt]#, full[excerpt]
        if not forever: 
            break

def extract_video_frames(input):
    # input should have shape
    # (b_s, seq_length, 3, 60, 120)
    sh = input.shape
    output = np.zeros((sh[0] * sh[1], sh[2], sh[3], sh[4]))
    for i in range(sh[0]):
        for j in range(sh[1]):
            output[i * sh[1] + j,:,:,:] = input[i,j,:,:,:]
    return output

def saveImage(imageData, path, side=8):

    # format data appropriately
    imageData = imageData.transpose(0,2,3,1).astype('uint8')

    #creates a new empty image, RGB mode, and size 400 by 400.
    new_im = Image.new('RGB', (160*side,80*side))
    
    index = 0
    for i in xrange(0,(side)*160,160):
        for j in xrange(0,(side)*80,80):
            #paste the image at location i,j:
            img = Image.fromarray(imageData[index])
            #img.show()
            new_im.paste(img, (i,j))
            index += 1

    new_im.save(path + '.png')

def save_model(model, model_name, epoch):
    np.savez('models/' + str(model_name) + '_' + str(epoch) + '.npz', *ll.get_all_param_values(model))

def load_model(model, model_name, epoch):
    param_path = 'models/' + str(model_name) + '_' + str(epoch) + '.npz'
    with np.load(param_path) as f:
        param_values = [f['arr_%d' % i] for i in range(len(f.files))]
        ll.set_all_param_values(model, param_values)     
    return model

'''
taken from the original repository 
https://github.com/commaai/research/blob/master/dask_generator.py
'''

def concatenate(camera_names, time_len):
    logs_names = [x.replace('camera', 'log') for x in camera_names]

    angle = [] # steering angle of the car
    speed = [] # steering angle of the car
    hdf5_camera = [] # the camera hdf5 files need to continue open
    c5x = []
    filters = []
    lastidx = 0

    for cword, tword in zip(camera_names, logs_names):
        try:
            with h5py.File(tword, "r") as t5:
                c5 = h5py.File(cword, "r")
                hdf5_camera.append(c5)
                x = c5["X"]
                c5x.append((lastidx, lastidx+x.shape[0], x))

                speed_value = t5["speed"][:]
                steering_angle = t5["steering_angle"][:]
                idxs = np.linspace(0, steering_angle.shape[0]-1, x.shape[0]).astype("int") # approximate alignment
                angle.append(steering_angle[idxs])
                speed.append(speed_value[idxs])

                goods = np.abs(angle[-1]) <= 200

                filters.append(np.argwhere(goods)[time_len-1:] + (lastidx+time_len-1))
                lastidx += goods.shape[0]
                # check for mismatched length bug
                print("x {} | t {} | f {}".format(x.shape[0], steering_angle.shape[0], goods.shape[0]))
                if x.shape[0] != angle[-1].shape[0] or x.shape[0] != goods.shape[0]:
                    raise Exception("bad shape")

        except IOError:
            import traceback
            traceback.print_exc()
            print "failed to open", tword

    angle = np.concatenate(angle, axis=0)
    speed = np.concatenate(speed, axis=0)
    filters = np.concatenate(filters, axis=0).ravel()
    print "training on %d/%d examples" % (filters.shape[0], angle.shape[0])
    return c5x, angle, speed, filters, hdf5_camera


first = True


def datagen(time_len=1, batch_size=64*500, ignore_goods=False):
    """
    Parameters:
    -----------
    leads : bool, should we use all x, y and speed radar leads? default is false, uses only x
    """
    global first
    global path
    assert time_len > 0

    files_in_cam_dir = [f for f in listdir(path + 'camera/') if isfile(join(path + 'camera/', f))]
    filter_files = [f for f in files_in_cam_dir if '.h5' in f]
    filter_files = [path + 'camera/' + x for x in filter_files]
    filter_names = sorted(filter_files)

    c5x, angle, speed, filters, hdf5_camera = concatenate(filter_names, time_len=time_len )
    filters_set = set(filters)

    X_batch = np.zeros((batch_size, time_len, 3, 160, 320), dtype='uint8')
    angle_batch = np.zeros((batch_size, time_len, 1), dtype='float32')
    speed_batch = np.zeros((batch_size, time_len, 1), dtype='float32')

    while True:
        try:
            t = time.time()

            count = 0
            start = time.time()
            while count < batch_size:
                if not ignore_goods:
                    i = np.random.choice(filters)
                    # check the time history for goods
                    good = True
                    for j in (i-time_len+1, i+1):
                        if j not in filters_set:
                            good = False
                    if not good:
                        continue

                else:
                    i = np.random.randint(time_len+1, len(angle), 1)

                # GET X_BATCH
                # low quality loop
                for es, ee, x in c5x:
                    if i >= es and i < ee:
                        X_batch[count] = x[i-es-time_len+1:i-es+1]
                        break

                angle_batch[count] = np.copy(angle[i-time_len+1:i+1])[:, None]
                speed_batch[count] = np.copy(speed[i-time_len+1:i+1])[:, None]

                count += 1

            # sanity check
            assert X_batch.shape == (batch_size, time_len, 3, 160, 320)

            print("%5.2f ms" % ((time.time()-start)*1000.0))

            if first:
                print "X", X_batch.shape
                print "angle", angle_batch.shape
                print "speed", speed_batch.shape
                first = False

            yield (X_batch, angle_batch, speed_batch)

        except KeyboardInterrupt:
            raise
        except Exception as e:
            #traceback.print_exc()
	    print e
pass


def cleanup_data(data, skip_frames, time_len, normalize=True):
    X = data[0]
    angle, speed = data[1], data[2]

    # remove extra frames
    X = X[:, ::skip_frames]
    sh = X.shape

    if time_len == 1 : 
        X = X.reshape((-1, 3, 160, 320))
        X = np.asarray([scipy.misc.imresize(x.transpose(1, 2, 0), (80, 160, 3)) for x in X])
        X = X.reshape(sh[0], 80, 160, 3) 
        X = X.transpose(0,3,1,2)
    else : 
        # we need to accomodate this extra dimension
        placeholder = np.zeros((sh[0], sh[1], sh[2], 80, 160))
        for i in range(sh[0]):
            for j in range(sh[1]):
                placeholder[i,j,:,:,:] = scipy.misc.imresize(X[i,j].transpose(1, 2, 0), (80, 160, 3)).transpose(2, 0, 1)
        X = placeholder

    X = (X / 255. - 0.5) / 0.5
    Z = np.concatenate([angle, speed], axis=-1)
    return Z, X

def data_iterator(batch_size, skip_frames=1, time_len=1):
    generator = datagen(batch_size=batch_size, time_len=time_len)
    for tup in generator:
        data = cleanup_data(tup, skip_frames, time_len=time_len)
    	yield data

def optimizer_factory(optimizer, grads, params, eta):
    if optimizer == 'rmsprop' : 
        return lasagne.updates.rmsprop(
            grads, params, learning_rate=eta)
    elif optimizer == 'adam' : 
        return lasagne.updates.adam(
            grads, params, learning_rate=eta)
    else : 
        raise Exception(optimizer + ' not supported')



