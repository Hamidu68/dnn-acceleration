import os, sys
import csv
import numpy as np

#import keras
#from keras.layers import Conv2D, MaxPooling2D, AveragePooling2D, BatchNormalization, ZeroPadding2D, Flatten, Dense, Activation
from keras import backend as K
from keras import Model
from keras import layers

#np.set_printoptions(threshold=np.inf, linewidth=9223372036854775807)

#######################################
#Keras layers functions
def add_InputLayer(info=None, fid=None, dtype=int):
    input_shape = eval(info['batch_input_shape'])
    #Read input value
    input_value = np.fromfile(file=fid, dtype=dtype, sep='', count=input_shape[1]*input_shape[2]*input_shape[3])
    input_value = input_value.reshape(input_shape[1:]).astype(np.float32)
    #Make input tensor
    input_tensor = layers.Input(shape=input_shape[1:])
    return input_value, input_tensor

def add_Conv2D(input_tensor=None, info=None, fid=None, dtype=int):
    #Read information of layer
    input_shape = eval(info['batch_input_shape'])
    output_shape = eval(info['batch_output_shape'])
    kernel_size = eval(info['kernel_size'])

    #Read weights from file
    weights=[]
    kernel = np.fromfile(file=fid, dtype=dtype, sep='', count=kernel_size[0]*kernel_size[1]*input_shape[3]*output_shape[3])
    kernel = kernel.reshape((kernel_size[0],kernel_size[1],input_shape[3],output_shape[3])).astype(np.float32)
    weights.append(kernel)
               
    if eval(info['use_bias']) == True:
        bias = np.fromfile(file=fid, dtype=dtype, sep='', count=output_shape[3])
        bias = bias.reshape((output_shape[3],)).astype(np.float32)
        weights.append(bias)

    #Get output tensor
    output_tensor = layer.Conv2D(filters= int(info['filters']),
                                 kernel_size= eval(info['kernel_size']),
                                 strides= eval(info['strides']),
                                 padding= str(info['padding']),
                                 data_format= str(info['data_format']),
                                 dilation_rate= eval(info['dilation_rate']),
                                 activation= str(info['activation']),
                                 use_bias= eval(info['use_bias']),
                                 weights=weights)(input_tensor)
    return output_tensor

def add_MaxPooling2D(input_tensor=None, info=None):
    #Get output tensor
    output_tensor = layers.MaxPooling2D(pool_size= eval(info['pool_size']),
                                        strides= eval(info['strides']),
                                        padding= str(info['padding']),
                                        data_format= str(info['data_format']))(input_tensor)
    return output_tensor

def add_BatchNormalization(input_tensor=None, info=None, fid=None, dtype=int):
    #Read information of layer
    output_shape = eval(info['batch_output_shape'])
    
    #Read weights from file
    weights = []
    for _ in range(4):  #Order: gamma, beta, running mean and running std
        temp_weights = np.fromfile(file=fid, dtype=dtype, sep='', count=output_shape[3])
        temp_weights = temp_weights.reshape((output_shape[3],)).astype(np.float32)
        weights.append(temp_weights)
    
    #Get output tensor
    output_tensor = layers.BatchNormalization(axis= int(info['axis']),
                                              momentum= float(info['momentum']),
                                              epsilon= float(info['epsilon']),
                                              center= eval(info['center']),
                                              scale= eval(info['scale']),
                                              weights=weights)(input_tensor)
    return output_tensor

def add_Activation(input_tensor=None, info=None):
    #Get output tensor
    output_tensor = layers.Activation(activation= str(info['activation']))(input_tensor)
    return output_tensor

def add_AveragePooling2D(input_tensor=None, info=None):
    #Get output tensor
    output_tensor = layers.AveragePooling2D(pool_size= eval(info['pool_size']),
                                            strides= eval(info['strides']),
                                            padding= str(info['padding']),
                                            data_format= str(info['data_format']))(input_tensor)
    return output_tensor

def add_ZeroPadding2D(input_tensor=None, info=None):
    #Get output tensor
    output_tensor = layers.ZeroPadding2D(padding= eval(info['padding']),
                                         data_format= str(info['data_format']))(input_tensor)
    return output_tensor

def add_Flatten(input_tensor=None, info=None):
    #Get output tensor
    output_tensor = layers.Flatten(data_format= str(info['data_format']))(input_tensor)
    return output_tensor

def add_Dense(input_tensor=None, info=None, fid=None, dtype=int):
    #Read information of layer
    input_shape = eval(info['batch_input_shape'])
    output_shape = eval(info['batch_output_shape'])

    #Read weights from file
    weights=[]
    nodes = np.fromfile(file=fid, dtype=dtype, sep='', count=input_shape[1]*output_shape[1])
    nodes = nodes.reshape((input_shape[1],output_shape[1])).astype(np.float32)
    weights.append(nodes)
               
    if eval(info['use_bias']) == True:
        bias = np.fromfile(file=fid, dtype=dtype, sep='', count=output_shape[1])
        bias = bias.reshape((output_shape[1],)).astype(np.float32)
        weights.append(bias)
    
    #Get output tensor
    output_tensor = layers.Dense(units= int(info['units']),
                                 activation= str(info['activation']),
                                 use_bias= eval(info['use_bias']),
                                 weights=weights)(input_tensor)
    return output_tensor

def add_Add(input_tensors=[], info=None):
    #Get output tensor
    output_tensor = layers.Add()(input_tensors)
    return output_tensor

#######################################
#Print function
def print_result(model=None, input_values=None):
    def printXD(ary, fid=None, shape=()):
        if len(shape) == 1:
            print1D(ary, fid, shape)
        elif len(shape) == 3:
            print3D(ary, fid, shape)
            
    def print1D(ary1D, fid=None, shape=()):
        fid.write('[[')
        for x in range(shape[0]):
            f.write('{:.6f} '.format(ary1D[x]))
        fid.write(']]')
        
    def print3D(ary3D, fid=None, shape=()):
        fid.write('[[')
        for x in range(shape[0]):
            fid.write('[')
            for y in range(shape[1]):
                fid.write('[')
                for z in range(shape[2]):
                    fid.write('{:.6f} '.format(ary3D[x][y][z]))
                if y != (shape[1] - 1):
                    fid.write(']\n   ')
                else:
                    fid.write(']')
            if x != (shape[0] - 1):
                fid.write(']\n\n  ')
        fid.write(']]]')
    
    #Print result
    print("[Keras_verifier.py]Print Result")
    
    #Open file
    f = open('Output/keras_output.txt','w')

    #Write values
    i = 0
    for layer in model.layers:
        layer_type = (str(layer).split()[0]).split('.')[-1]

        f.write('{} : '.format(layer_type))
        if layer_type == 'InputLayer':
            #Write input values
            printXD(input_values[i], f, input_values[i].shape))

        else:
            #Get output values of each layer
            get_3rd_layer_output = K.function([model.layers[0].input], [layer.output])
            layer_output = get_3rd_layer_output([input_values])(0)
            #Write output values
            printXD(layer_output[0], f, layer_output[0].shape))
        f.write('\n\n')
    
    #model.summary()
    f.close()
    

#######################################
#Functions about input_tensors & outputs_dict
def get_single_input(inbound_name='', tensors={}, outputs_dict={}):
    #get output tensor of inbound layer
    input_tensor = tensors[inbound_name]
    
    #delete output of inbound layer in outputs_dict
    if inbound_name in outputs_dict:
        del outputs_dict[inbound_name]
    
    return input_tensor

def get_multi_inputs(inbound_names=[], tensors={}, outputs_dict={}):
    #get output tensors of inbound layers
    input_tensors = []

    for name in inbound_names:
        input_tensors.append(tensors[name])
        
        #delete outputs of inbound layers in outputs_dict
        if name in outputs_dict:
            del outputs_dict[name]
    
    return input_tensors

#######################################
#main
if __name__ == "__main__":

    #Read csv file
    csv_file=open(sys.argv[1])
    csv_reader=csv.DictReader(csv_file)

    #Read weight and input file
    weights_bin=open(sys.argv[2],'rb')
    inputs_bin=open(sys.argv[3],'rb')

    #Check data type of weight and input files
    dtype = np.int32
    dtype2 = "int"
    if sys.argv[4] == 'int':
        dtype = np.int32
        dtype2 = "int"
    elif sys.argv[4] == 'unsigned int':
        dtype = np.uint32
        dtype2 = "int"
    elif sys.argv[4] == 'float':
        dtype = np.float32
        dtype2 = "float"
    elif sys.argv[4] == 'ap_uint<16>':
        dtype = np.uint16
        dtype2 = "int"
    else:
        print('Wrong data type!')
    
    #Skip layers
    skip_layers = ['Dropout']

    #init parameters
    line_num=-1
    input_values= np.array([])
    inputs=[]
    outputs=[]
    outputs_dict={}
    tensors = {}
    
    #for each layers
    for row in csv_reader:
        #check skip layer
        layer_type=row["layer_type"]
        if layer_type in skip_layers:
            print('Skip {} layer')
            continue
        
        line_num=line_num+1
        layer_name=row['name']
        print('[Keras_verifier.py]Operate ' + layer_type + '.' + str(line_num))

        #Switch
        if layer_type == 'InputLayer':
            input_value, tensors[layer_name] = add_InputLayer(row, inputs_bin, dtype)
            input_values.append(input_value)
            inputs.append(tensors[layer_name])            
            
        elif layer_type == 'Conv2D':
            #get input tensor
            input_tensor = get_single_input(row['connected_to'], tensors, outputs_dict)
            #get output of current layer and save it to dict
            tensors[layer_name] = add_Conv2D(input_tensor, row, weights_bin, dtype)
            #append output of current layer to outputs_dict
            outputs_dict[layer_name] = tensors[layer_name]
            
        elif layer_type == 'MaxPooling2D':
            #get input tensor
            input_tensor = get_single_input(row['connected_to'], tensors, outputs_dict)
            #get output of current layer and save it to dict
            tensors[layer_name] = add_MaxPooling2D(input_tensor, row)
            #append output of current layer to outputs_dict
            outputs_dict[layer_name] = tensors[layer_name]
            
        elif layer_type == 'BatchNormalization':
            #get input tensor
            input_tensor = get_single_input(row['connected_to'], tensors, outputs_dict)
            #get output of current layer and save it to dict
            tensors[layer_name] = add_BatchNormalization(input_tensor, row, weights_bin, dtype)
            #append output of current layer to outputs_dict
            outputs_dict[layer_name] = tensors[layer_name]
                        
        elif layer_type == 'Activation':
            #get input tensor
            input_tensor = get_single_input(row['connected_to'], tensors, outputs_dict)
            #get output of current layer and save it to dict
            tensors[layer_name] = add_Activation(input_tensor, row)
            #append output of current layer to outputs_dict
            outputs_dict[layer_name] = tensors[layer_name]
            
        elif layer_type == 'AveragePooling2D':
            #get input tensor
            input_tensor = get_single_input(row['connected_to'], tensors, outputs_dict)
            #get output of current layer and save it to dict
            tensors[layer_name] = add_AveragePooling2D(input_tensor, row)
            #append output of current layer to outputs_dict
            outputs_dict[layer_name] = tensors[layer_name]
            
        elif layer_type == 'ZeroPadding2D':
            #get input tensor
            input_tensor = get_single_input(row['connected_to'], tensors, outputs_dict)
            #get output of current layer and save it to dict
            tensors[layer_name] = add_ZeroPadding2D(input_tensor, row)
            #append output of current layer to outputs_dict
            outputs_dict[layer_name] = tensors[layer_name]

        elif layer_type == 'Flatten':
            #get input tensor
            input_tensor = get_single_input(row['connected_to'], tensors, outputs_dict)
            #get output of current layer and save it to dict
            tensors[layer_name] = add_Flatten(input_tensor, row)
            #append output of current layer to outputs_dict
            outputs_dict[layer_name] = tensors[layer_name]
            
        elif layer_type == 'Dense':
            #get input tensor
            input_tensor = get_single_input(row['connected_to'], tensors, outputs_dict)
            #get output of current layer and save it to dict
            tensors[layer_name] = add_Dense(input_tensor, row, weights_bin, dtype)
            #append output of current layer to outputs_dict
            outputs_dict[layer_name] = tensors[layer_name]
            
        elif layer_type == 'Add':
            #get input tensors
            input_tensors = get_multi_inputs(row['connected_to'].split('/'), tensors, outputs_dict)
            #get output of current layer and save it to dict
            tensors[layer_name] = add_Add(input_tensors, row)
            #append output of current layer to outputs_dict
            outputs_dict[layer_name] = tensors[layer_name]
            
        else:
            print('Undefined Layer: {}'.format(layer_type))
    
    
    #Build model
    for output in outputs_dict.values():
        outputs.append(output)
    model = Model(inputs=inputs, outputs=outputs)
    temp = model.predict(input_values)


    #Print the result
    print_result(model, input_values)


    #Close weight and input binary files
    weights_bin.close()
    inputs_bin.close()
