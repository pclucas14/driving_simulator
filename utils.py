import numpy as np
import h5py
import scipy.misc
from PIL import Image

def load_dataset(normalize=True, resize=True, sample=False):
    path = '/home/ml/lpagec/research/dataset/camera/'

    # check if processed file in memory : 
    try : 
        f = file(path + 'data.bin', "rb")
        print "found cached dataset"
        dataset = np.load(f)
        f.close()
        return dataset
    except : 
        print "no cached version. processing raw images."
        camera_file = h5py.File(path + '2016-02-02--10-16-58.h5')
        camera_file = camera_file['X']
        camera_file = np.array(camera_file)
        if sample : camera_file = cmaera_file[:1000]
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
        
        # remove random images at the beginning
        if not sample : 
            dataset = dataset[900:]
            dataset = dataset[:-1500]
            np.random.shuffle(dataset)
        dataset = dataset.astype('float32')
        dataset /= 255.
        
        # put dataset in Theano ordering
        if dataset.shape[1] != 3 : 
            dataset = dataset.transpose(0, 3, 1, 2)

        if normalize : 
            dataset -= 0.5
            dataset /= 0.5
        
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

def saveImage(imageData, epoch, side=8):

    # format data appropriately
    imageData = imageData.transpose(0,2,3,1).astype('uint8')

    #creates a new empty image, RGB mode, and size 400 by 400.
    new_im = Image.new('RGB', (80*side,160*side))
    
    # imageData = imageData.reshape((-1,64,64,3))
    
    #Iterate through a 4 by 4 grid with 100 spacing, to place my image
    index = 0
    for i in xrange(0,(side)*80,80):
        for j in xrange(0,(side)*160,160):
            #paste the image at location i,j:
            img = Image.fromarray(imageData[index])
            #img.show()
            new_im.paste(img, (i,j))
            index += 1

    new_im.save('sample_epoch' + str(epoch) + '.png')
    #new_im.save(home + 'images/' + imageName+ '_epoch' + str(epoch) + '.png')
