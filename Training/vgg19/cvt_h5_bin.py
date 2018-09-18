
import os, sys
import numpy as np

#keras
from keras.models import load_model

def cvt_h5_bin(h5_path = './weights_h5',
               bin_path = './weights_bin',
               model_name = 'vgg19',
               epoch = 0):
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
    '''
    #functions
    def weights1D(weights=None):
        for m in range(weights.shape[0]):
            f.write(weights[m])

    def weights2D(weights=None):
        for m in range(weights.shape[0]):
            for k in range(weights.shape[1]):
                f.write(weights[m][k])

    def weights3D(weights=None):
        for m in range(weights.shape[0]):
            for k in range(weights.shape[1]):
                for i in range(weights.shape[2]):
                    f.write(weights[m][k][i])

    def weights4D(weights=None):
        for m in range(weights.shape[0]):
            for k in range(weights.shape[1]):
                for i in range(weights.shape[2]):
                    for j in range(weights.shape[3]):
                        f.write(weights[m][k][i][j])
    '''
    
    #Load the binary file
    f = open(bin_path + '/' + model_name + '_weights.bin', 'wb')
    
    #Save weights as binary file
    def save_weights(layers=None):
        for layer in model.layers:
            if (str(layer).split()[0]).split('.')[-1] == 'Model':
                save_weights(layer)
            else:
                layer.get_weights().tofile(f)
        '''
        for W in layer.weights:
            print(W.shape)
            print(type(W))
            #f.write(bytearray(np.asarray(W)))
           
            if len(W.shape) == 4:
                weights4D(W)
            elif len(W.shape) == 3:
                weights3D(W)
            elif len(W.shape) == 2:
                weights2D(W)
            elif len(W.shape) == 1:
                weights1D(W)
        '''

    #Close the binary file
    f.close()


if __name__ == '__main__':
    cvt_h5_bin()
