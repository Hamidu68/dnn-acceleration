from .keras_layers import *


def keras_generator(model_data, model_name, dtype_str, paths, skip_layers):
    output_path = paths[0]
    template_path = paths[1]
    weight_file_path = paths[2]
    input_file_path = paths[3]

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

    # init parameters
    line_num = -1
    input_values = np.array([])
    inputs = []
    outputs = []
    outputs_dict = {}
    tensors = {}

    # for each layers
    for layer in model_data.layers:
        row = layer.config

        layer_type = row["layer_type"]
        layer_name = row['name']
        previous_row = row['connected_to']

        # Check skip layer
        if layer_type in skip_layers:
            skip = True
        else:
            skip = False
            line_num=line_num+1
            print('[keras_verifier.py]Calculate ' + layer_type + str(line_num))

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
    print_keras(model, input_values, name, output_path)

    # Close weight and input binary files
    weights_bin.close()
    inputs_bin.close()


def print_keras(model=None, input_values=None, name=None, output_path =''):
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
    print("[keras_verifier.py]Print Result")

    # Open file
    f = open(output_path+'/output_value/keras_output.txt', 'w')
    fn = open(output_path+'/output_value/keras_output_num.txt', 'w')

    # Write values
    i = 0
    for layer in model.layers:
        layer_type = (str(layer).split()[0]).split('.')[-1]

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
