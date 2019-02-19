from .Keras_Layers import *
from .Print_Keras import *


def Keras_Verifier(model_data, model_name, weight_file_path, input_file_path, dtype_str):
    # Read weight and input file
    weights_bin=open(weight_file_path, 'rb')
    inputs_bin=open(input_file_path, 'rb')

    # Check data type of weight and input files
    dtype = np.int32
    if dtype_str == 'int':
        dtype = np.int32
    elif dtype_str == 'unsigned int':
        dtype = np.uint32
    elif dtype_str == 'float':
        dtype = np.float32
    elif dtype_str == 'ap_uint<16>':
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
    previous_row = ''
    
    # for each layers
    for layer in model_data.layers:
        row = layer.config
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
        previous_row = row['connected_to']
        
        # Switch
        if layer_type == 'InputLayer':
            input_value, tensors[layer_name] = add_InputLayer(row, inputs_bin, dtype)
            input_values = input_value
            inputs.append(tensors[layer_name])

        elif layer_type == 'Conv2D':
            # get input tensor
            input_tensor = get_single_input(previous_row, tensors, outputs_dict)
            # get output of current layer and save it to dict
            outputs_dict[layer_name] = tensors[layer_name] = add_Conv2D(input_tensor, row, weights_bin, dtype, skip, tensors)

        elif layer_type == 'DepthwiseConv2D':
            # get input tensor
            input_tensor = get_single_input(previous_row, tensors, outputs_dict)
            # get output of current layer and save it to dict
            outputs_dict[layer_name] = tensors[layer_name] = add_DepthConv2D(input_tensor, row, weights_bin, dtype, skip, tensors)

        elif layer_type == 'SeparableConv2D':
            # get input tensor
            input_tensor = get_single_input(previous_row, tensors, outputs_dict)
            # get output of current layer and save it to dict
            outputs_dict[layer_name] = tensors[layer_name] = add_SepConv2D(input_tensor, row, weights_bin, dtype, skip, tensors)

        elif layer_type == 'MaxPooling2D':
            # get input tensor
            input_tensor = get_single_input(previous_row, tensors, outputs_dict)
            # get output of current layer and save it to dict
            outputs_dict[layer_name] = tensors[layer_name] = add_MaxPooling2D(input_tensor, row, skip, tensors)

        elif layer_type == 'BatchNormalization':
            # get input tensor
            input_tensor = get_single_input(previous_row, tensors, outputs_dict)
            # get output of current layer and save it to dict
            outputs_dict[layer_name] = tensors[layer_name] = add_BatchNormalization(input_tensor, row, weights_bin, dtype, skip, tensors)

        elif layer_type == 'ReLU':
            # get input tensor
            input_tensor = get_single_input(previous_row, tensors, outputs_dict)
            # get output of current layer and save it to dict
            outputs_dict[layer_name] = tensors[layer_name] = add_ReLU(input_tensor, row, skip, tensors)

        elif layer_type == 'Activation':
            # get input tensor
            input_tensor = get_single_input(previous_row, tensors, outputs_dict)
            # get output of current layer and save it to dict
            outputs_dict[layer_name] = tensors[layer_name] = add_Activation(input_tensor, row, skip, tensors)

        elif layer_type == 'Reshape':
            input_tensor = get_single_input(previous_row, tensors, outputs_dict)
            # get output of current layer and save it to dict
            outputs_dict[layer_name] = tensors[layer_name] = add_Reshape(input_tensor, row, skip, tensors)

        elif layer_type == 'AveragePooling2D':
            # get input tensor
            input_tensor = get_single_input(previous_row, tensors, outputs_dict)
            # get output of current layer and save it to dict
            outputs_dict[layer_name] = tensors[layer_name] = add_AveragePooling2D(input_tensor, row, skip, tensors)

        elif layer_type == 'ZeroPadding2D':
            # get input tensor
            input_tensor = get_single_input(previous_row, tensors, outputs_dict)
            # get output of current layer and save it to dict
            outputs_dict[layer_name] = tensors[layer_name] = add_ZeroPadding2D(input_tensor, row, skip, tensors)

        elif layer_type == 'Flatten':
            # get input tensor
            input_tensor = get_single_input(previous_row, tensors, outputs_dict)
            # get output of current layer and save it to dict
            outputs_dict[layer_name] = tensors[layer_name] = add_Flatten(input_tensor, row, skip, tensors)

        elif layer_type == 'Dense':
            # get input tensor
            input_tensor = get_single_input(previous_row, tensors, outputs_dict)
            # get output of current layer and save it to dict
            outputs_dict[layer_name] = tensors[layer_name] = add_Dense(input_tensor, row, weights_bin, dtype, skip, tensors)

        elif layer_type == 'Dropout':
            # get input tensor
            input_tensor = get_single_input(previous_row, tensors, outputs_dict)
            # get output of current layer and save it to dict
            outputs_dict[layer_name] = tensors[layer_name] = add_Dropout(input_tensor, row, skip, tensors)

        elif layer_type == 'Add':
            # get input tensors
            input_tensors = get_multi_inputs(previous_row.split('/'), tensors, outputs_dict)
            # get output of current layer and save it to dict
            outputs_dict[layer_name] = tensors[layer_name] = add_Add(input_tensors, row, skip, tensors)

        elif layer_type == 'Concatenate':
            # get input tensors
            input_tensors = get_multi_inputs(previous_row.split('/'), tensors, outputs_dict)
            # get output of current layer and save it to dict
            outputs_dict[layer_name] = tensors[layer_name] = add_Concatenate(input_tensors, row, skip, tensors)

        elif layer_type == 'Lambda':
            # get input tensors
            input_tensors = get_multi_inputs(previous_row.split('/'), tensors, outputs_dict)
            # get output of current layer and save it to dict
            outputs_dict[layer_name] = tensors[layer_name] = add_Lambda(input_tensors, row, skip, tensors)

        elif layer_type == 'GlobalAveragePooling2D':
            # get input tensor
            input_tensor = get_single_input(previous_row, tensors, outputs_dict)
            # get output of current layer and save it to dict
            outputs_dict[layer_name] = tensors[layer_name] = add_GlobalAveragePooling2D(input_tensor, row, skip, tensors)

        elif layer_type == 'GlobalMaxPooling2D':
            # get input tensor
            input_tensor = get_single_input(previous_row, tensors, outputs_dict)
            # get output of current layer and save it to dict
            outputs_dict[layer_name] = tensors[layer_name] = add_GlobalMaxPooling2D(input_tensor, row, skip, tensors)

        elif layer_type == 'Cropping2D':
            # get input tensor
            input_tensor = get_single_input(previous_row, tensors, outputs_dict)
            # get output of current layer and save it to dict
            outputs_dict[layer_name] = tensors[layer_name] = add_Cropping2D(input_tensor, row, skip, tensors)

        else:
            print('Undefined Layer: {}'.format(layer_type))

    # Build model
    for output in outputs_dict.values():
        outputs.append(output)

    model = Model(inputs=inputs, outputs=outputs)
    temp = model.predict(input_values)
    name = model_name

    # Print the result
    Print_Keras(model, input_values, name)

    # Close weight and input binary files
    weights_bin.close()
    inputs_bin.close()

