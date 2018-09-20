
import os, sys
import numpy as np

#keras
from keras.models import load_model

def cvt_h5_bin(h5_path = './weights_h5',
               bin_path = './weights_bin',
               model_name = 'vgg19',
               epoch = 0,
               dtype=np.float32):
    '''
    '''
    #Load the model
    if epoch != 0:
        model = load_model(h5_path + '/' + model_name + '_weights.' + str(epoch) + '.h5')
    else:
        file_name=''
        for file in os.listdir(h5_path):
            file_name_list = file.split('.')
            if file_name_list[-1] == 'h5':
                file_name=file
        model = load_model(h5_path + '/' + file_name)
    
    #Load the binary file
    f = open(bin_path + '/' + model_name + '_weights.bin', 'wb')
    
    #Save weights to binary file
    def save_weights(layers=None, fid=None, dtype=np.float32):
        for layer in layers:
            if (str(layer).split()[0]).split('.')[-1] == 'Model':
                save_weights(layer.layers, fid, dtype)
            else:
                for W in layer.get_weights():
                    W.astype(dtype).tofile(fid)

    save_weights(model.layers, f, dtype)

    #Close the binary file
    f.close()


if __name__ == '__main__':
    cvt_h5_bin(h5_path = './weights_h5',
               bin_path = './weights_bin',
               model_name = 'vgg19',
               epoch = 0,
               dtype=np.float32)
