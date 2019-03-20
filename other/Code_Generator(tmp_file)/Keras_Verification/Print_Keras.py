import os, sys
import csv
import numpy as np

from keras import backend as K

# Print function
def Print_Keras(model=None, input_values=None, name=None):
    np.set_printoptions(threshold=np.inf)
    np.set_printoptions(linewidth=999999999999999999999999999999)

    def printXD(ary, fid=None, fn=None, shape=()):
        if len(shape) == 1:
            print1D(ary, fid, fn, shape)
        elif len(shape) == 3:
            print3D(ary, fid, fn, shape)

    def print1D(ary1D, fid=None, fn=None, shape=()):
        fid.write('[[')
        for x in range(shape[0]):
            fid.write('{:.6f} '.format(ary1D[x]))
            fn.write('{:.6f} '.format(ary1D[x]))
        fid.write(']]\n\n')

    def print3D(ary3D, fid=None, fn=None, shape=()):
        fid.write('[[')
        for x in range(shape[0]):
            fid.write('[')
            for y in range(shape[1]):
                fid.write('[')
                for z in range(shape[2]):
                    fid.write('{:.6f} '.format(ary3D[x][y][z]))
                    fn.write('{:.6f} '.format(ary3D[x][y][z]))
                if y != (shape[1] - 1):
                    fid.write(']\n   ')
                else:
                    fid.write(']')
            if x != (shape[0] - 1):
                fid.write(']\n\n  ')
        fid.write(']]]\n\n')

    # Print result
    print("[Keras_verifier.py]Print Result")

    # Open file
    f = open('Produced_code/'+name+'/Output/keras_output.txt', 'w')
    fn = open('Produced_code/'+name+'/Output/keras_output_num.txt', 'w')

    # Write values
    i = 0
    for layer in model.layers:
        layer_type = (str(layer).split()[0]).split('.')[-1]

        skip_layers = ['Dropout']
        if layer_type in skip_layers:
            skip = True
            continue

        f.write('{} : '.format(layer_type))
        if layer_type == 'InputLayer':
            # Write input values
            printXD(input_values[i], f, fn, input_values[i].shape)

        else:
            # Get output values of each layer
            get_3rd_layer_output = K.function([model.layers[0].input], [layer.output])
            layer_output = get_3rd_layer_output([input_values])[0]
            # Write output values
            printXD(layer_output[0], f, fn, layer_output[0].shape)

    # model.summary()
    f.close()
