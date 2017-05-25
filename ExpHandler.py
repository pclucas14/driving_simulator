import os 
import datetime
from utils import * 

'''
Class responsible for storing all data structures related to an experiment
'''
class ExpHandler():
    def __init__(self, params_dict, dum_freq=100, test=False):
        self.params = params_dict
        self.params['date'] = datetime.datetime.now()
        self.epoch = 0
        self.dum_freq = dum_freq

        # initialize a directory in experiments
        files = [f for f in listdir('experiments/') if not isfile(join('experiments/', f))]
        files = [x.replace('exp_', '') for x in files]
        files = [int(x) for x in files]
        files = sorted(files)
        index = int(files[-1]) + 1 if len(files) > 0 else 0
        print index
        self.exp_name = 'experiments/exp_' + str(index)
        # print 'exp path : ', self.exp_name

        # make directory : 
        os.makedirs(self.exp_name)
        os.makedirs(self.exp_name + '/images')

        # store parameters to file
        with open (self.exp_name + '/params.txt', 'w') as fp:
            for p in self.params.items():
                fp.write("%s : %s\n" % p)

        self.monitored_models = dict()

    def save_image(self, samples, real_img=False, extra=""):
        if self.params['image_prepro'] == 'DCGAN' : 
            samples *= 0.5; samples += 0.5; samples *= 255.
        else : 
            raise Exception('unsupported image preprocessing')
        if real_img : 
            saveImage(samples, self.exp_name + '/images' + '/real_' + str(self.epoch))
        else : 
            saveImage(samples, self.exp_name + '/images' + '/' + extra + 'samples_' + str(self.epoch))

    def add_model(self, name, layers, monitored_values):
        aModel = Model(name, layers, monitored_values, self.exp_name)
        self.monitored_models[aModel.name] = aModel

        if self.params['load_weights'] is not None : 
            exp_no, epoch = self.params['load_weights']
            aModel.load_params(exp_no, epoch)

    def record(self, model_name, values):
        self.monitored_models[model_name].values.append(values)

    def end_of_epoch(self):
        self.epoch += 1
        # to know which experiment you are running every time an epoch is printed
        print self.exp_name
        if self.epoch % self.dum_freq == 0 : 
            for model in self.monitored_models.values():
                model.dump_values()
                model.save_params(self.epoch)


'''
Class to encapsulate a few nice methods for model logging.
This class should -ideally- only be used by ExpHandler.  
'''
class Model():
    def __init__(self, name, network, monitored_values, path):
        self.name = name
        self.network = network # last layer to be specific

        # basically the output of a training function. 
        self.monitored_values = monitored_values 

        # ExpHandler is responsible for setting appropriate path
        # path should be experiments/exp_XX/model_name
        self.path = path + '/' + name

        # make params directory
        os.makedirs(self.path)
        os.makedirs(self.path + '/weights')

        self.firstDump = True
        self.values = []
        self.params = monitored_values

        # store parameters to file
        with open (path + '/' + str(name) + '/params.txt', 'w') as fp:
            for p in self.params.items():
                fp.write("%s : %s\n" % p)
            fp.write('model architecture : \n')
            for layer in network : 
                fp.write(str(layer.output_shape) + '\n')


    def save_params(self, epoch):
        np.savez(self.path +  '/weights/' + str(epoch) + '.npz', *ll.get_all_param_values(self.network[-1]))


    def load_params(self, exp_no, epoch):
        try : 
            param_path = 'experiments/exp_' + str(exp_no) + '/' + self.name + '/weights/' + str(epoch) + '.npz'
            with np.load(param_path) as f:
                param_values = [f['arr_%d' % i] for i in range(len(f.files))]
                ll.set_all_param_values(self.network[-1], param_values)   
            print 'loaded weights for model ' + self.name 
        except Exception as e : 
            print 'weights NOT loaded for model ' + self.name
            print 'the following error occured : '
            print e


    def dump_values(self):
        if self.firstDump : 
            # create file
            with open (self.path + '/monitoring.txt', 'w') as f: 
                # first write all the observed parameters on one line
                first_line = ""
                for param_name in self.monitored_values.keys():
                    first_line += param_name + ',     '
                first_line = str(first_line)
                f.write(first_line)
                f.close()
            self.firstDump = False
        
        # dump values to file
        with open (self.path + '/monitoring.txt', 'a') as f: 
            # first write all the observed parameters on one line
            for line in self.values : 
                f.write('\n' + str(line))
            f.close()
        self.values = []




        

