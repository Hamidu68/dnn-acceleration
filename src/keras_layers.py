import os, sys
import numpy as np
import ast

from keras import backend as K
K.set_image_dim_ordering('th')

from keras import Model
from keras import layers
from keras.engine.topology import get_source_inputs

#import tensorflow as tf
#from tensorflow.keras import layers

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


def add_Conv2D(input_tensor=None, info=None, fid=None, dtype=int, skip=False, tensors = {}):
    if skip:
        return tensors[info['connected_to']]

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


def add_MaxPooling2D(input_tensor=None, info=None, skip=False, tensors = {}):
    if skip:
        return tensors[info['connected_to']]

    # Get output tensor
    output_tensor = layers.MaxPooling2D(pool_size=eval(info['pool_size']),
                                        strides=eval(info['strides']),
                                        padding=str(info['padding']),
                                        data_format=str(info['data_format']))(input_tensor)
    return output_tensor


def add_Cropping2D(input_tensor=None, info=None, skip=False, tensors = {}):
    if skip:
        return tensors[info['connected_to']]

    # Get output tensor
    output_tensor = layers.Cropping2D(cropping=eval(info['cropping']), data_format=str(info['data_format']))(input_tensor)
    return output_tensor


def add_BatchNormalization(input_tensor=None, info=None, fid=None, dtype=int, skip=False, tensors = {}):
    if skip:
        return tensors[info['connected_to']]

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


def add_Activation(input_tensor=None, info=None, skip=False, tensors = {}):
    if skip:
        return tensors[info['connected_to']]

    # Get output tensor
    output_tensor = layers.Activation(activation=str(info['activation']))(input_tensor)
    return output_tensor


def add_AveragePooling2D(input_tensor=None, info=None, skip=False, tensors = {}):
    if skip:
        return tensors[info['connected_to']]

    # Get output tensor
    output_tensor = layers.AveragePooling2D(pool_size=eval(info['pool_size']),
                                            strides=eval(info['strides']),
                                            padding=str(info['padding']),
                                            data_format=str(info['data_format']))(input_tensor)
    return output_tensor


def add_ZeroPadding2D(input_tensor=None, info=None, skip=False, tensors = {}):
    if skip:
        return tensors[info['connected_to']]

    # Get output tensor
    output_tensor = layers.ZeroPadding2D(padding=eval(info['padding']),
                                         data_format=str(info['data_format']))(input_tensor)
    return output_tensor


def add_Flatten(input_tensor=None, info=None, skip=False, tensors = {}):
    if skip:
        return tensors[info['connected_to']]

    # Get output tensor
    output_tensor = layers.Flatten(data_format=str(info['data_format']))(input_tensor)
    return output_tensor


def add_Dropout(input_tensor=None, info=None, skip=False, tensors = {}):
    if skip:
        return tensors[info['connected_to']]

    # Get output tensor
    output_tensor = layers.Dropout(rate=eval(info['rate']))(input_tensor)
    return output_tensor


def add_Lambda(input_tensor=None, info=None, skip=False, tensors = {}):
    if skip:
        return tensors[info['connected_to']]

    input_shape = eval(info['batch_input_shape'])
    output_shape = eval(info['batch_output_shape'])
    scale = ast.literal_eval(info['arguments'])

    if input_tensor is not None:
        inputs = get_source_inputs(input_tensor)

    # Get output tensor
    output_tensor = layers.Lambda(lambda input_shape, scale: inputs[0] + inputs[1] * scale,
                                  output_shape=output_shape[1:],
                                  arguments=scale)(input_tensor)
    return output_tensor


def add_DepthConv2D(input_tensor=None, info=None, fid=None,  dtype=int, skip=False, tensors = {}):
    if skip:
        return tensors[info['connected_to']]

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


def add_SepConv2D(input_tensor=None, info=None, fid=None,  dtype=int, skip=False, tensors = {}):
    if skip:
        return tensors[info['connected_to']]

    # Read information of layer
    output_shape = eval(info['batch_output_shape'])
    input_shape = eval(info['batch_input_shape'])
    kernel_size = eval(info['kernel_size'])

    # Read weights from file
    weights = []
    weight1 = np.fromfile(file=fid, dtype=dtype, sep='',
                          count=kernel_size[0] * kernel_size[1] * input_shape[3])
    weight1 = weight1.reshape((kernel_size[0], kernel_size[1], input_shape[3])).astype(np.float32)
    weights.append(weight1)

    weight2 = np.fromfile(file=fid, dtype=dtype, sep='',
                          count=output_shape[3] * input_shape[3])
    weight2 = weight2.reshape((input_shape[3], output_shape[3],)).astype(np.float32)
    weights.append(weight2)

    # Get output tensor
    output_tensor = layers.SeparableConv2D(kernel_size=kernel_size,
                                           strides=eval(info['strides']),
                                           padding=str(info['padding']),
                                           depth_multiplier=1,
                                           dilation_rate=eval(info['dilation_rate']),
                                           data_format=str(info['data_format']),
                                           activation=str(info['activation']),
                                           use_bias=str(info['use_bias']),
                                           weights=weights
                                           )(input_tensor)
    return output_tensor


def add_Dense(input_tensor=None, info=None, fid=None, dtype=int, skip=False, tensors = {}):
    if skip:
        return tensors[info['connected_to']]

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


def add_ReLU(input_tensor=None, info=None, skip=False, tensors = {}):
    if skip:
        return tensors[info['connected_to']]

    # Get output tensor
    output_tensor = layers.ReLU(max_value=int(eval(info['max_value'])))(input_tensor)
    return output_tensor


def add_Add(input_tensors=[], info=None, skip=False, outputs_dict = {}):
    if skip:
        return tensors[info['connected_to']]

    # Get output tensor
    output_tensor = layers.Add()(input_tensors)
    return output_tensor


def add_Reshape(input_tensors=[], info=None, skip=False, tensors = {}):
    if skip:
        return tensors[info['connected_to']]

    # Read information of layer
    output_shape = eval(info['batch_output_shape'])

    # Get output tensor
    output_tensor = layers.Reshape(target_shape=output_shape[1:])(input_tensors)
    return output_tensor


def add_Concatenate(input_tensors=[], info=None, skip=False, tensors = {}):
    if skip:
        return tensors[info['connected_to']]

    # Get output tensor
    output_tensor = layers.Concatenate(axis=-1)(input_tensors)
    return output_tensor


def add_GlobalAveragePooling2D(input_tensors=[], info=None, skip=False, tensors = {}):
    if skip:
        return tensors[info['connected_to']]

    # Get output tensor
    output_tensor = layers.GlobalAveragePooling2D(data_format=str(info['data_format']))(input_tensors)
    return output_tensor


def add_GlobalMaxPooling2D(input_tensors=[], info=None, skip=False, tensors = {}):
    if skip:
        return tensors[info['connected_to']]

    # Get output tensor
    output_tensor = layers.GlobalMaxPooling2D(data_format=str(info['data_format']))(input_tensors)
    return output_tensor
