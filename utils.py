import numpy as np
import h5py
import scipy.misc
from PIL import Image
from model import * 
from os import listdir
from os.path import isfile, join
import time
import lasagne

path = '/NOBACKUP/dash_cam_dataset/'


def load_dataset_dummy():
    return np.ones((1000, 3, 80, 160))


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

def optimizer_factory(optimizer, grads, params, eta):
    if optimizer == 'rmsprop' : 
        return lasagne.updates.rmsprop(
            grads, params, learning_rate=eta)
    elif optimizer == 'adam' : 
        return lasagne.updates.adam(
            grads, params, learning_rate=eta)
    else : 
        raise Exception(optimizer + ' not supported')

def format_imgs(samples, flatten=False):
    samples *= 0.5; samples += 0.5; samples *= 255.
    samples.astype('uint8')
    if len(samples.shape) == 5:
        if flatten: 
            samples = extract_video_frames(samples)
            samples = samples.transpose(0,2,3,1).astype('uint8')
        else :
            samples = samples.transpose(0,1,3,4,2).astype('uint8')
    else :
        samples = samples.transpose(0,2,3,1).astype('uint8')
    return samples






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
    angle = angle[::skip_frames]
    speed = speed[::skip_frames]

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



