import os, sys
import csv
import numpy as np

# import keras
# from keras.layers import Conv2D, MaxPooling2D, AveragePooling2D, BatchNormalization, ZeroPadding2D, Flatten, Dense, Activation

from keras import backend as K
from keras import Model
from keras import layers

#import tensorflow as tf
#from tensorflow.keras import layers


np.set_printoptions(threshold=np.inf, linewidth=922337203685477)

####################################################
####################################################
# Functions about input_tensors & outputs_dict


def get_single_input(inbound_name='', tensors={}, outputs_dict={}):
    # get output tensor of inbound layer
    input_tensor = tensors[inbound_name]
    
    # delete output of inbound layer in outputs_dict
    if inbound_name in outputs_dict:
        del outputs_dict[inbound_name]
    
    return input_tensor


def get_multi_inputs(inbound_names=[], tensors={}, outputs_dict={}):
    # get output tensors of inbound layers
    input_tensors = []

    for name in inbound_names:
        input_tensors.append(tensors[name])
        
        # delete outputs of inbound layers in outputs_dict
        if name in outputs_dict:
            del outputs_dict[name]
    
    return input_tensors


####################################################
# Keras layers functions
def add_InputLayer(info=None, fid=None, dtype=int):
    input_shape = eval(info['batch_input_shape'])
    # Read input value
    input_value = np.fromfile(file=fid, dtype=dtype, sep='', count=input_shape[1]*input_shape[2]*input_shape[3])
    input_value = input_value.reshape((1, input_shape[1], input_shape[2], input_shape[3])).astype(np.float32)
    # Make input tensor
    input_tensor = layers.Input(shape=input_shape[1:])
    return input_value, input_tensor


def add_Conv2D(input_tensor=None, info=None, fid=None, dtype=int):
    # Read information of layer
    input_shape = eval(info['batch_input_shape'])
    output_shape = eval(info['batch_output_shape'])
    kernel_size = eval(info['kernel_size'])

    # Read weights from file
    weights = []
    kernel = np.fromfile(file=fid, dtype=dtype, sep='', count=kernel_size[0]*kernel_size[1]*input_shape[3]*output_shape[3])
    kernel = kernel.reshape((kernel_size[0], kernel_size[1], input_shape[3], output_shape[3])).astype(np.float32)
    weights.append(kernel)
               
    if eval(info['use_bias']):
        bias = np.fromfile(file=fid, dtype=dtype, sep='', count=output_shape[3])
        bias = bias.reshape((output_shape[3],)).astype(np.float32)
        weights.append(bias)

    # Get output tensor
    output_tensor = layers.Conv2D(filters=int(info['filters']),
                                  kernel_size=eval(info['kernel_size']),
                                  strides=eval(info['strides']),
                                  padding=str(info['padding']),
                                  data_format=str(info['data_format']),
                                  dilation_rate=eval(info['dilation_rate']),
                                  activation=str(info['activation']),
                                  use_bias=eval(info['use_bias']),
                                  weights=weights)(input_tensor)
    return output_tensor


def add_MaxPooling2D(input_tensor=None, info=None):
    # Get output tensor
    output_tensor = layers.MaxPooling2D(pool_size=eval(info['pool_size']),
                                        strides=eval(info['strides']),
                                        padding=str(info['padding']),
                                        data_format=str(info['data_format']))(input_tensor)
    return output_tensor


def add_Cropping2D(input_tensor=None, info=None):
    # Get output tensor
    output_tensor = layers.MaxPooling2D(cropping=eval(info['cropping']), dim_ordering='tf')(input_tensor)
    return output_tensor


def add_BatchNormalization(input_tensor=None, info=None, fid=None, dtype=int, skip=False):
    if skip:
        return input_tensor

    # Read information of layer
    output_shape = eval(info['batch_output_shape'])
    
    # Read weights from file
    weights = []

    if info['scale'] == 'False':
        for _ in range(3):  # Order: gamma, beta, running mean and running std
            temp_weights = np.fromfile(file=fid, dtype=dtype, sep='', count=output_shape[3])
            temp_weights = temp_weights.reshape((output_shape[3],)).astype(np.float32)
            weights.append(temp_weights)
    else:
        for _ in range(4):  # Order: gamma, beta, running mean and running std
            temp_weights = np.fromfile(file=fid, dtype=dtype, sep='', count=output_shape[3])
            temp_weights = temp_weights.reshape((output_shape[3],)).astype(np.float32)
            weights.append(temp_weights)
    
    # Get output tensor
    output_tensor = layers.BatchNormalization(axis=int(info['axis']),
                                              momentum=float(info['momentum']),
                                              epsilon=float(info['epsilon']),
                                              center=eval(info['center']),
                                              scale=eval(info['scale']),
                                              weights=weights)(input_tensor)
    return output_tensor


def add_Activation(input_tensor=None, info=None):
    # Get output tensor
    output_tensor = layers.Activation(activation=str(info['activation']))(input_tensor)
    return output_tensor


def add_AveragePooling2D(input_tensor=None, info=None):
    # Get output tensor
    output_tensor = layers.AveragePooling2D(pool_size=eval(info['pool_size']),
                                            strides=eval(info['strides']),
                                            padding=str(info['padding']),
                                            data_format=str(info['data_format']))(input_tensor)
    return output_tensor


def add_ZeroPadding2D(input_tensor=None, info=None):
    # Get output tensor
    output_tensor = layers.ZeroPadding2D(padding=eval(info['padding']),
                                         data_format=str(info['data_format']))(input_tensor)
    return output_tensor


def add_Flatten(input_tensor=None, info=None):
    # Get output tensor
    output_tensor = layers.Flatten(data_format=str(info['data_format']))(input_tensor)
    return output_tensor


def add_Dropout(input_tensor=None, info=None):
    # Get output tensor
    output_tensor = layers.Dropout(rate=eval(info['rate']))(input_tensor)
    return output_tensor


def add_Lambda(input_tensor=None, info=None):
    # Get output tensor
    output_tensor = layers.Lambda()(input_tensors)
    return output_tensor


def add_DepthConv2D(input_tensor=None, info=None, fid=None,  dtype=int):
    # Read information of layer
    output_shape = eval(info['batch_output_shape'])
    kernel_size = eval(info['kernel_size'])

    # Read weights from file
    weights = []
    kernel = np.fromfile(file=fid, dtype=dtype, sep='',
                         count=kernel_size[0] * kernel_size[1] * output_shape[3])
    kernel = kernel.reshape((kernel_size[0], kernel_size[1], output_shape[3], 1)).astype(np.float32)
    weights.append(kernel)

    # Get output tensor
    output_tensor = layers.DepthwiseConv2D(kernel_size=eval(info['kernel_size']),
                                           strides=eval(info['strides']),
                                           padding=str(info['padding']),
                                           data_format=str(info['data_format']),
                                           activation=str(info['activation']),
                                           use_bias=eval(info['use_bias']),
                                           weights=weights
                                           )(input_tensor)
    return output_tensor


def add_SeparableConv2D(input_tensor=None, info=None, fid=None,  dtype=int):
    # Read information of layer
    output_shape = eval(info['batch_output_shape'])
    input_shape = eval(info['batch_input_shape'])
    kernel_size = eval(info['kernel_size'])

    # Read weights from file
    weight1 = []
    weight2 = []
    kernel1 = np.fromfile(file=fid, dtype=dtype, sep='',
                          count=kernel_size[0] * kernel_size[1] * input_shape[3])
    kernel1 = kernel1.reshape((kernel_size[0], kernel_size[1], input_shape[3])).astype(np.float32)
    weight1.append(kernel1)

    kernel2 = np.fromfile(file=fid, dtype=dtype, sep='',
                          count=output_shape[3] * input_shape[3])
    kernel2 = kernel2.reshape((input_shape[3], output_shape[3])).astype(np.float32)
    weight2.append(kernel2)

    # Get output tensor
    output_tensor = layers.SeparableConv2D(kernel_size=eval(info['kernel_size']),
                                           strides=eval(info['strides']),
                                           padding=str(info['padding']),
                                           depth_multiplier=1,
                                           dilation_rate=eval(info['dilation_rate']),
                                           data_format=str(info['data_format']),
                                           activation=str(info['activation']),
                                           use_bias=eval(info['use_bias']),
                                           weights=[weight1, weight2]
                                           )(input_tensor)
    return output_tensor


def add_Dense(input_tensor=None, info=None, fid=None, dtype=int):
    # Read information of layer
    input_shape = eval(info['batch_input_shape'])
    output_shape = eval(info['batch_output_shape'])

    # Read weights from file
    weights=[]
    nodes = np.fromfile(file=fid, dtype=dtype, sep='', count=input_shape[1]*output_shape[1])
    nodes = nodes.reshape((input_shape[1], output_shape[1])).astype(np.float32)
    weights.append(nodes)
               
    if eval(info['use_bias']):
        bias = np.fromfile(file=fid, dtype=dtype, sep='', count=output_shape[1])
        bias = bias.reshape((output_shape[1],)).astype(np.float32)
        weights.append(bias)
    
    # Get output tensor
    output_tensor = layers.Dense(units=int(info['units']),
                                 activation=str(info['activation']),
                                 use_bias=eval(info['use_bias']),
                                 weights=weights)(input_tensor)
    return output_tensor


def add_ReLU(input_tensor=None, info=None):
    # Get output tensor
    output_tensor = layers.ReLU(max_value=int(eval(info['max_value'])))(input_tensor)
    return output_tensor


def add_Add(input_tensors=[], info=None):
    # Get output tensor
    output_tensor = layers.Add()(input_tensors)
    return output_tensor


def add_Reshape(input_tensors=[], info=None):
    # Read information of layer
    output_shape = eval(info['batch_output_shape'])

    # Get output tensor
    output_tensor = layers.Reshape(target_shape=output_shape[1:])(input_tensors)
    return output_tensor


def add_Concatenate(input_tensors=[], info=None):
    # Get output tensor
    output_tensor = layers.Concatenate(axis=-1)(input_tensors)
    return output_tensor


def add_GlobalAveragePooling2D(input_tensors=[], info=None):
    # Get output tensor
    output_tensor = layers.GlobalAveragePooling2D(data_format=str(info['data_format']))(input_tensors)
    return output_tensor


####################################################
####################################################
# Print function
def print_result(model=None, input_values=None, name=None):
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
    f = open('../cpp_generator/'+name+'/Output/keras_output.txt', 'w')
    fn = open('../cpp_generator/'+name+'/Output/keras_output_num.txt', 'w')

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


####################################################
####################################################
# main
if __name__ == "__main__":


    # Read csv file
    csv_file=open(sys.argv[1])
    csv_reader=csv.DictReader(csv_file)

    # Read weight and input file
    weights_bin=open(sys.argv[2], 'rb')
    inputs_bin=open(sys.argv[3], 'rb')

    # Check data type of weight and input files
    dtype = np.int32
    if sys.argv[4] == 'int':
        dtype = np.int32
    elif sys.argv[4] == 'unsigned int':
        dtype = np.uint32
    elif sys.argv[4] == 'float':
        dtype = np.float32
    elif sys.argv[4] == 'ap_uint<16>':
        dtype = np.uint16
    else:
        print('Wrong data type!')
    
    # Skip layers
    skip_layers = ['Dropout']

    # init parameters
    line_num = -1
    input_values = np.array([])
    inputs = []
    outputs = []
    outputs_dict = {}
    tensors = {}
    
    # for each layers
    for row in csv_reader:
        # check skip layer
        layer_type = row["layer_type"]
        if layer_type in skip_layers:
            skip = True
            print('Skip {} layer')
        else:
            skip = False
            line_num=line_num+1
            print('[Keras_verifier.py]add a layer: ' + layer_type + '.' + str(line_num))
        
        layer_name = row['name']

        # Switch
        if layer_type == 'InputLayer':
            input_value, tensors[layer_name] = add_InputLayer(row, inputs_bin, dtype)
            input_values = input_value
            inputs.append(tensors[layer_name])
            
        elif layer_type == 'Conv2D':
            # get input tensor
            input_tensor = get_single_input(row['connected_to'], tensors, outputs_dict)
            # get output of current layer and save it to dict
            outputs_dict[layer_name] = tensors[layer_name] = add_Conv2D(input_tensor, row, weights_bin, dtype)

        elif layer_type == 'DepthwiseConv2D':
            # get input tensor
            input_tensor = get_single_input(row['connected_to'], tensors, outputs_dict)
            # get output of current layer and save it to dict
            outputs_dict[layer_name] = tensors[layer_name] = add_DepthConv2D(input_tensor, row, weights_bin, dtype)

        elif layer_type == 'SeparableConv2D':
            # get input tensor
            input_tensor = get_single_input(row['connected_to'], tensors, outputs_dict)
            # get output of current layer and save it to dict
            outputs_dict[layer_name] = tensors[layer_name] = add_SeparableConv2D(input_tensor, row, weights_bin, dtype)

        elif layer_type == 'MaxPooling2D':
            # get input tensor
            input_tensor = get_single_input(row['connected_to'], tensors, outputs_dict)
            # get output of current layer and save it to dict
            outputs_dict[layer_name] = tensors[layer_name] = add_MaxPooling2D(input_tensor, row)
            
        elif layer_type == 'BatchNormalization':
            # get input tensor
            input_tensor = get_single_input(row['connected_to'], tensors, outputs_dict)
            # get output of current layer and save it to dict
            outputs_dict[layer_name] = tensors[layer_name] = add_BatchNormalization(input_tensor, row, weights_bin, dtype, skip)

        elif layer_type == 'ReLU':
            # get input tensor
            input_tensor = get_single_input(row['connected_to'], tensors, outputs_dict)
            # get output of current layer and save it to dict
            outputs_dict[layer_name] = tensors[layer_name] = add_ReLU(input_tensor, row)
                        
        elif layer_type == 'Activation':
            # get input tensor
            input_tensor = get_single_input(row['connected_to'], tensors, outputs_dict)
            # get output of current layer and save it to dict
            outputs_dict[layer_name] = tensors[layer_name] = add_Activation(input_tensor, row)

        elif layer_type == 'Reshape':
            input_tensor = get_single_input(row['connected_to'], tensors, outputs_dict)
            # get output of current layer and save it to dict
            outputs_dict[layer_name] = tensors[layer_name] = add_Reshape(input_tensor, row)
            
        elif layer_type == 'AveragePooling2D':
            # get input tensor
            input_tensor = get_single_input(row['connected_to'], tensors, outputs_dict)
            # get output of current layer and save it to dict
            outputs_dict[layer_name] = tensors[layer_name] = add_AveragePooling2D(input_tensor, row)
            
        elif layer_type == 'ZeroPadding2D':
            # get input tensor
            input_tensor = get_single_input(row['connected_to'], tensors, outputs_dict)
            # get output of current layer and save it to dict
            outputs_dict[layer_name] = tensors[layer_name] = add_ZeroPadding2D(input_tensor, row)

        elif layer_type == 'Flatten':
            # get input tensor
            input_tensor = get_single_input(row['connected_to'], tensors, outputs_dict)
            # get output of current layer and save it to dict
            outputs_dict[layer_name] = tensors[layer_name] = add_Flatten(input_tensor, row)
            
        elif layer_type == 'Dense':
            # get input tensor
            input_tensor = get_single_input(row['connected_to'], tensors, outputs_dict)
            # get output of current layer and save it to dict
            outputs_dict[layer_name] = tensors[layer_name] = add_Dense(input_tensor, row, weights_bin, dtype)

        elif layer_type == 'Dropout':
            # get input tensor
            input_tensor = get_single_input(row['connected_to'], tensors, outputs_dict)
            # get output of current layer and save it to dict
            outputs_dict[layer_name] = tensors[layer_name] = add_Dropout(input_tensor, row)
            
        elif layer_type == 'Add':
            # get input tensors
            input_tensors = get_multi_inputs(row['connected_to'].split('/'), tensors, outputs_dict)
            # get output of current layer and save it to dict
            outputs_dict[layer_name] = tensors[layer_name] = add_Add(input_tensors, row)

        elif layer_type == 'Concatenate':
            # get input tensors
            input_tensors = get_multi_inputs(row['connected_to'].split('/'), tensors, outputs_dict)
            # get output of current layer and save it to dict
            outputs_dict[layer_name] = tensors[layer_name] = add_Concatenate(input_tensors, row)

        elif layer_type == 'Lambda':
            # get input tensors
            input_tensors = get_multi_inputs(row['connected_to'].split('/'), tensors, outputs_dict)
            # get output of current layer and save it to dict
            outputs_dict[layer_name] = tensors[layer_name] = add_Lambda(input_tensors, row)

        elif layer_type == 'GlobalAveragePooling2D':
            # get input tensor
            input_tensor = get_single_input(row['connected_to'], tensors, outputs_dict)
            # get output of current layer and save it to dict
            outputs_dict[layer_name] = tensors[layer_name] = add_GlobalAveragePooling2D(input_tensor, row)

        elif layer_type == 'Cropping2D':
            # get input tensor
            input_tensor = get_single_input(row['connected_to'], tensors, outputs_dict)
            # get output of current layer and save it to dict
            outputs_dict[layer_name] = tensors[layer_name] = add_Cropping2D(input_tensor, row)
            
        else:
            print('Undefined Layer: {}'.format(layer_type))

    # Build model
    for output in outputs_dict.values():
        outputs.append(output)

    model = Model(inputs=inputs, outputs=outputs)
    temp = model.predict(input_values)
    name = sys.argv[5]

    # Print the result
    print_result(model, input_values, name)

    # Close weight and input binary files
    weights_bin.close()
    inputs_bin.close()

    c_out = open("../cpp_generator/"+name+"/Output/c_output_num.txt", 'r')
    k_out = open("../cpp_generator/"+name+"/Output/keras_output_num.txt", 'r')

    c_output_line = c_out.readline()
    c = c_output_line.split()

    k_output_line = k_out.readline()
    k = k_output_line.split()

    maximum = -1.0
    c_max = 0
    k_max = 0

    print
    "len k :  ", len(k)
    print
    "len c :  ", len(c)

    for i in range(len(c)):
        k_num = float(k[i])
        c_num = float(c[i])
        if c_num > k_num:
            if k_num == 0.0:
                error = c_num - k_num
            else:
                error = (c_num - k_num) / k_num
        else:
            if k_num == 0.0:
                error = k_num - c_num
            else:
                error = (k_num - c_num) / k_num

        if maximum < error:
            maximum = error
            c_max = c_num
            k_max = k_num
    print("maximum error : " + str(maximum) + " when c has an element of " + str(c_max) +
          " and keras has an element of " + str(k_max))

    c_out.close()
    k_out.close()