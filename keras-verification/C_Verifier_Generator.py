import os
from sys import argv
import csv
from string import Template

####################################################
# layer_type = InputLayer
def add_InputLayer(row=0, line_count=0, templates={}):
    output_name = 'I'
    result = ['','','','','','']

    input_shape = eval(row['batch_input_shape'])

    #SW_static_variables (I)
    result[2] = "static DATA_T {}[{}][{}][{}];\n\t".format(output_name,input_shape[1],input_shape[2],input_shape[3])
    
    #Initialization
    result[3] = templates['Init']['3D'].substitute({'third':input_shape[1],     #k
                                                    'second':input_shape[2],    #i
                                                    'first':input_shape[3],     #j
                                                    'iter3':'k', 'iter2':'i', 'iter1':'j',
                                                    'fid':'i_stream',
                                                    'ary_name':output_name}) + "\n\t"
    #print result
    result[5] = templates['Print']['3D'].substitute({'Name':row['layer_type'],
                                                     'third':input_shape[1],
                                                     'second':input_shape[2],
                                                     'first':input_shape[3],
                                                     'output':output_name}) + "\n"
    return output_name, result

# layer_type = convolution2D(I,O,B,W) (padding option: valid , same) (other: relu, bias, stride)
def add_Conv2D(row=0, line_count=0, outputs_dict={}, templates={}):
    input_name = outputs_dict[row['connected_to']]
    output_name = 'O{}_SW'.format(line_count)
    result = ['','','','','','']

    input_shape = eval(row['batch_input_shape'])
    output_shape = eval(row['batch_output_shape'])
    kernel_size = eval(row['kernel_size'])
    strides = eval(row['strides'])

    #Shared_variables(W,B)
    weight_name = 'W{}'.format(line_count)
    result[1] = "static DATA_T {}[{}][{}][{}][{}];\n\t".format(weight_name,output_shape[3],input_shape[3],kernel_size[0],kernel_size[1])
    if eval(row['use_bias']) == True:
        bias_name = 'B{}'.format(line_count)
        result[1] += "static DATA_T {}[{}];\n\t".format(bias_name,output_shape[3])
    
    #SW_static_variables(O)
    result[2] = "static DATA_T {}[{}][{}][{}];\n\t".format(output_name,output_shape[1],output_shape[2],output_shape[3])

    #Initialization
    result[3] = templates['Init']['4D'].substitute({'fourth':kernel_size[0],    #m
                                                    'third':kernel_size[1],     #k
                                                    'second':input_shape[3],    #i
                                                    'first':output_shape[3],     #j
                                                    'iter4':'j', 'iter3':'i', 'iter2':'m', 'iter1':'k',
                                                    'fid':'w_stream',
                                                    'ary_name':weight_name}) + "\n\t"
    if eval(row['use_bias']) == True:
        result[3] = template['Init']['1D'].substitute({'first':output_shape[3], #j
                                                       'iter1':'j',
                                                       'fid':'w_stream',
                                                       'ary_name':bias_name}) + '\n\t'

    #Function use
    result[4] = "SW_{}({},{},{}".format(row['name'],input_name,output_name,weight_name)
    if eval(row['use_bias']) == True:
        result[4] += ',{}'.format(bias_name)
    result[4] += ');\n\t'

    #print result
    result[5] = templates['Print']['3D'].substitute({'Name':row['layer_type'],
                                                     'third':output_shape[1],
                                                     'second':output_shape[2],
                                                     'first':output_shape[3],
                                                     'output':output_name}) + "\n\t"

    #function def (Padding option)
    subs = {'Name' : row["name"], 'Input_channel' : input_shape[1], 'Input_width' : input_shape[2], 'Input_height' : input_shape[2],
            'Output_channel' : output_shape[1],'Output_width' : output_shape[2], 'Output_height' : output_shape[3],
            'Filter_width' : kernel_size[0], 'Filter_height' : kernel_size[1],
            'Stride_width': strides[0],'Stride_height':strides[1]}
    if row['padding'] == 'valid':
        result[0] = templates['Conv2D']['valid'].substitute(subs) +"\n\t"
    elif row['padding'] == 'same':
        if row['activation'] == 'relu':
            result[0] = templates['Conv2D']['same']['relu'].substitute(subs) +"\n\t"
        elif row['activation'] == 'linear':
            result[0] = templates['Conv2D']['same']['linear'].substitute(subs) +"\n\t"
    
    return output_name, result

# layer_type = MaxPooling2D (I, O)
def add_MaxPooling2D(row=0, line_count=0, outputs_dict={}, templates={}):       
    input_name = outputs_dict[row['connected_to']]
    output_name = 'O{}_SW'.format(line_count)
    result = ['','','','','','']

    input_shape = eval(row['batch_input_shape'])
    output_shape = eval(row['batch_output_shape'])
    pool_size = eval(row['pool_size'])
    strides = eval(row['strides'])

    #Shared_variables
    #SW_static_variables(O)
    result[2] = "static DATA_T {}[{}][{}][{}];\n\t".format(output_name,output_shape[1],output_shape[2],output_shape[3])
    #Initialization
    #Function use
    result[4] = "SW_{}({},{})".format(row['name'],input_name,output_name)
    #print result
    result[5] = templates['Print']['3D'].substitute({'Name':row['layer_type'],
                                                     'third':output_shape[1],
                                                     'second':output_shape[2],
                                                     'first':output_shape[3],
                                                     'output':output_name}) + "\n\t"
    #function def
    subs = {'Name' : row["name"], 'Input_channel': input_shape[1], 'Input_width': input_shape[2], 'Input_height': input_shape[3],
            'Output_channel': output_shape[1], 'Output_width': output_shape[2], 'Output_height': output_shape[3],
            'Stride_width': strides[0], 'Stride_height': strides[1],
            'Pool_width': pool_size[0], 'Pool_height': pool_size[1]}
    result[0] = templates['MaxPooling2D'].substitute(subs) + "\n\t"
    return output_name, result

# layer_type = BatchNormalization(I, O)
# if Batch size = None(one Input), Mean = 0, Var = 1 fixed.
def add_BatchNormalization(row=0, line_count=0, outputs_dict={}, templates={}, skip=False):
    input_name = outputs_dict[row['connected_to']]
    output_name = 'O{}_SW'.format(line_count)
    result = ['','','','','','']

    input_shape = eval(row['batch_input_shape'])
    output_shape = eval(row['batch_output_shape'])
    
    #Shared_variables
    #SW_static_variables(O)
    result[2] = "static DATA_T {}[{}][{}][{}];\n\t".format(output_name,output_shape[1],output_shape[2],output_shape[3])
    #Initialization
    #Function use
    result[4] = "SW_{}({},{})".format(row['name'],input_name,output_name)
    #print result
    result[5] = templates['Print']['3D'].substitute({'Name':row['layer_type'],
                                                     'third':output_shape[1],
                                                     'second':output_shape[2],
                                                     'first':output_shape[3],
                                                     'output':output_name}) + "\n\t"
    #function def
    subs = {'Name' : row["name"], 'Input_channel': input_shape[1], 'Input_width': input_shape[2], 'Input_height': input_shape[3],
            'Output_channel': output_shape[1], 'Output_width': output_shape[2], 'Output_height': output_shape[3]}
    result[0] = templates['BatchNormalization'].substitute(subs) + '\n\t'
    
    return output_name, result

# layer_type = Activation(relu)(I, O)
def add_Activation(row=0, line_count=0, outputs_dict={}, templates={}):
    input_name = outputs_dict[row['connected_to']]
    output_name = 'O{}_SW'.format(line_count)
    result = ['','','','','','']

    input_shape = eval(row['batch_input_shape'])
    output_shape = eval(row['batch_output_shape'])
    
    #Shared_variables
    #SW_static_variables(O)
    result[2] = "static DATA_T {}[{}][{}][{}];\n\t".format(output_name,output_shape[1],output_shape[2],output_shape[3])
    #Initialization
    #Function use
    result[4] = "SW_{}({},{})".format(row['name'],input_name,output_name)
    #print result
    result[5] = templates['Print']['3D'].substitute({'Name':row['layer_type'],
                                                     'third':output_shape[1],
                                                     'second':output_shape[2],
                                                     'first':output_shape[3],
                                                     'output':output_name}) + "\n\t"
    #function def
    subs = {'Name' : row["name"], 'Input_channel': input_shape[1], 'Input_width': input_shape[2], 'Input_height': input_shape[3],
            'Output_channel': output_shape[1], 'Output_width': output_shape[2], 'Output_height': output_shape[3]}
    if row['activation'] == 'relu':
        result[0] = templates['Activation']['relu'].substitute(subs) + '\n\t'
    elif row['activation'] == 'softmax':
        result[0] = templates['Activation']['softmax'].substitute(subs) + '\n\t'
        
    return output_name, result

# layer_type = AveragePooling2D (I, O)
def add_AveragePooling2D(row=0, line_count=0, outputs_dict={}, templates={}):            
    input_name = outputs_dict[row['connected_to']]
    output_name = 'O{}_SW'.format(line_count)
    result = ['','','','','','']

    input_shape = eval(row['batch_input_shape'])
    output_shape = eval(row['batch_output_shape'])
    pool_size = eval(row['pool_size'])
    strides = eval(row['strides'])
    
    #Shared_variables
    #SW_static_variables(O)
    result[2] = "static DATA_T {}[{}][{}][{}];\n\t".format(output_name,output_shape[1],output_shape[2],output_shape[3])
    #Initialization
    #Function use
    result[4] = "SW_{}({},{})".format(row['name'],input_name,output_name)
    #print result
    result[5] = templates['Print']['3D'].substitute({'Name':row['layer_type'],
                                                     'third':output_shape[1],
                                                     'second':output_shape[2],
                                                     'first':output_shape[3],
                                                     'output':output_name}) + "\n\t"
    #function def
    subs = {'Name' : row["name"], 'Input_channel': input_shape[1], 'Input_width': input_shape[2], 'Input_height': input_shape[3],
            'Output_channel': output_shape[1], 'Output_width': output_shape[2], 'Output_height': output_shape[3],
            'Stride_width': strides[0], 'Stride_height': strides[1],
            'Pool_width': pool_size[0], 'Pool_height': pool_size[1]}
    result[0] = templates['AveragePooling2D'].substitute(subs) + '\n\t'
    
    return output_name, result

# layer_type = ZeroPadding2D(I,O)
def add_ZeroPadding2D(row=0, line_count=0, outputs_dict={}, templates={}): 
    input_name = outputs_dict[row['connected_to']]
    output_name = 'O{}_SW'.format(line_count)
    result = ['','','','','','']

    input_shape = eval(row['batch_input_shape'])
    output_shape = eval(row['batch_output_shape'])
    padding = eval(row['padding'])
    
    #Shared_variables
    #SW_static_variables(O)
    result[2] = "static DATA_T {}[{}][{}][{}];\n\t".format(output_name,output_shape[1],output_shape[2],output_shape[3])
    #Initialization
    #Function use
    result[4] = "SW_{}({},{})".format(row['name'],input_name,output_name)
    #print result
    result[5] = templates['Print']['3D'].substitute({'Name':row['layer_type'],
                                                     'third':output_shape[1],
                                                     'second':output_shape[2],
                                                     'first':output_shape[3],
                                                     'output':output_name}) + "\n\t"
    #function def
    subs = {'Name' : row["name"], 'Input_channel': input_shape[1], 'Input_width': input_shape[2], 'Input_height': input_shape[3],
            'Output_channel': output_shape[1], 'Output_width': output_shape[2], 'Output_height': output_shape[3],
            'Padding_size': padding[0][0]}
    result[0] = templates['ZeroPadding2D'].substitute(subs) + '\n\t'
    
    return output_name, result

# layer_type = Flatten(I,O)
def add_Flatten(row=0, line_count=0, outputs_dict={}, templates={}):
    input_name = outputs_dict[row['connected_to']]
    output_name = 'O{}_SW'.format(line_count)
    result = ['','','','','','']

    input_shape = eval(row['batch_input_shape'])
    output_shape = eval(row['batch_output_shape'])
    
    #Shared_variables
    #SW_static_variables(O)
    result[2] = "static DATA_T {}[{}];\n\t".format(output_name,output_shape[1])
    #Initialization
    #Function use
    result[4] = "SW_{}({},{})".format(row['name'],input_name,output_name)
    #print result
    result[5] = templates['Print']['1D'].substitute({'Name':row['layer_type'],
                                                     'first':output_shape[1],
                                                     'output':output_name}) + "\n\t"
    #function def
    subs = {'Name' : row["name"], 'Input_channel': input_shape[1], 'Input_width': input_shape[2], 'Input_height': input_shape[3],
            'Output_channel': output_shape[1]}
    result[0] = templates['Flatten'].substitute(subs) + '\n\t'
    
    return output_name, result

# layer_type = Dense(I,W,B,O) (Activation option : relu , softmax)
def add_Dense(row=0, line_count=0, outputs_dict={}, templates={}):       
    input_name = outputs_dict[row['connected_to']]
    output_name = 'O{}_SW'.format(line_count)
    result = ['','','','','','']

    input_shape = eval(row['batch_input_shape'])
    output_shape = eval(row['batch_output_shape'])
    
    #Shared_variables
    weight_name = 'W{}'.format(line_count)
    result[1] = "static DATA_T {}[{}][{}];\n\t".format(weight_name,output_shape[1],input_shape[1])
    if eval(row['use_bias']) == True:
        bias_name = 'B{}'.format(line_count)
        result[1] += "static DATA_T {}[{}];\n\t".format(bias_name,output_shape[1])
    
    #SW_static_variables(O)
    result[2] = "static DATA_T {}[{}];\n\t".format(output_name,output_shape[1])

    #Initialization
    result[3] = templates['Init']['2D'].substitute({'second':input_shape[1],    #i
                                                    'first':output_shape[1],     #j
                                                    'iter2':'j', 'iter1':'i',
                                                    'fid':'w_stream',
                                                    'ary_name':weight_name}) + "\n\t"
    if eval(row['use_bias']) == True:
        result[3] = template['Init']['1D'].substitute({'first':output_shape[1], #j
                                                       'iter1':'j',
                                                       'fid':'w_stream',
                                                       'ary_name':bias_name}) + '\n\t'
    
    #Function use
    result[4] = "SW_{}({},{},{}".format(row['name'],input_name,output_name,weight_name)
    if eval(row['use_bias']) == True:
        result[4] += ',{}'.format(bias_name)
    result[4] += ');\n\t'
    
    #print result
    result[5] = templates['Print']['1D'].substitute({'Name':row['layer_type'],
                                                     'first':output_shape[1],
                                                     'output':output_name}) + "\n\t"
    
    #function def
    subs = {'Name': row["name"], 'Input_channel': input_shape[1], 'Output_channel': output_shape[1]}
    if row['activation'] == 'relu':
        result[0] = templates['Dense']['relu'].substitute(subs) + '\n\t'
    elif row['activation'] == 'softmax':
        result[0] = templates['Dense']['softmax'].substitute(subs) + '\n\t'
    
    return output_name, result

# layer_type = Add(I1,I2,O)
def add_Add(row=0, line_count=0, outputs_dict={}, templates={}):
    input_names = [outputs_dict[name] for name in row['connected_to'].split('/')]
    output_name = 'O{}_SW'.format(line_count)
    result = ['','','','','','']

    input_shape1 = eval(row['batch_input_shape'])[0]
    input_shape2 = eval(row['batch_input_shape'])[1]
    output_shape = eval(row['batch_output_shape'])
    
    #Shared_variables
    #SW_static_variables(O)
    result[2] = "static DATA_T {}[{}][{}][{}];\n\t".format(output_name,output_shape[1],output_shape[2],output_shape[3])
    #Initialization
    #Function use
    result[4] = "SW_{}({},{},{})".format(row['name'],input_names[0],input_names[1],output_name)
    #print result
    result[5] = templates['Print']['3D'].substitute({'Name':row['layer_type'],
                                                     'third':output_shape[1],
                                                     'second':output_shape[2],
                                                     'first':output_shape[3],
                                                     'output':output_name}) + "\n\t"
    #function def
    subs = {'Name' : row["name"], 'Input_channel1': input_shape1[1], 'Input_width1': input_shape1[2], 'Input_height1': input_shape1[3],
            'Input_channel2': input_shape2[1], 'Input_width2': input_shape2[2], 'Input_height2': input_shape2[3],
            'Output_channel': output_shape[1], 'Output_width': output_shape[2], 'Output_height': output_shape[3]}
    result[0] = templates['Add'].substitute(subs) + '\n\t'
    
    return output_name, result


####################################################
#main
#sys.argv[1]= Test.csv
#sys.argv[2]= Data_type
if __name__ == "__main__":
    # Main 0) Check data type of weight and input files
    # int, unsigned int, float, ap_uint<16>
    if argv[2] == 'ap_uint<16>':
        dtype = 'unsigned short'
    else:
        dtype = argv[2]
    
    
    # Main 1) Define variable
    
    SW_def_func = ""
    Shared_static_v = ""
    SW_static_v = ""
    initialization = ""
    SW_functions = ""
    print_result = ""

    # Main 2) Load Template

    templates = {}
    templates['Main'] = Template(open("../Template/Main/C_Verification.txt").read())
    templates['Init'] = {'1D':Template(open("../Template/Init/Init1D.txt").read()),
                         '2D':Template(open("../Template/Init/Init2D.txt").read())
                         '3D':Template(open("../Template/Init/Init3D.txt").read()),
                         '4D':Template(open("../Template/Init/Init4D.txt").read())}
    templates['Print'] = {'1D':Template(open("../Template/Print/Print_Output1D.txt").read()),
                          '3D':Template(open("../Template/Print/Print_Output3D.txt").read())}
    templates['Conv2D'] = {'valid':Template(open("../Template/Function/Conv2D_valid.txt").read()),
                           'same':{'linear':Template(open("../Template/Function/Conv2D_same.txt").read()),
                                   'relu':Template(open("../Template/Function/Conv2D_same_relu.txt").read())}}
    templates['MaxPooling2D'] = Template(open("../Template/Function/MaxPooling2D.txt").read())
    templates['BatchNormalization'] = Template(open("../Template/Function/BatchNormalization.txt").read())
    templates['Activation'] = {'relu':Template(open("../Template/Function/Relu.txt").read()),
                               'softmax':Template(open("../Template/Function/Softmax.txt").read())}
    templates['AveragePooling2D'] = Template(open("../Template/Function/AveragePooling2D.txt").read())
    templates['ZeroPadding2D'] = Template(open("../Template/Function/ZeroPadding.txt").read())
    templates['Flatten'] = Template(open("../Template/Function/Flatten.txt").read())
    templates['Dense'] = {'relu':Template(open("../Template/Function/Dense_Relu.txt").read()),
                          'softmax':Template(open("../Template/Function/Dense_Softmax.txt").read())}
    templates['Add'] = Template(open("../Template/Function/Add.txt").read())

    #not implemented
    templates['GlobalMaxPooling'] = Template(open("../Template/Function/GlobalMaxPooling.txt").read())
    templates['GlobalAveragePooling'] = Template(open("../Template/Function/GlobalAveragePooling.txt").read())
    
    # Main 3) Read Layer Information from CSV

    csv_file = open(argv[1])
    csv_reader = csv.DictReader(csv_file)

    # Main 4) Generate Function depending on layer_type

    #Skip layers
    skip_layers = ['Dropout']
    
    line_count = -1
    outputs_dict={}
    
    #for each layers
    for row in csv_reader:
        #check skip layer
        layer_type=row["layer_type"]
        if layer_type in skip_layers:
            skip = True
        else:
            skip = False
            #Count Line number
            line_count+= 1

        layer_name=row['name']
        
        #Report Status
        SW_functions += "printf(\"[C_verifier.cpp]{}.{}\\n\");\n\t".format(layer_type,line_count)

        result = ['','','','','','']
        
        #Switch
        if layer_type == 'InputLayer':
            outputs_dict[layer_name], result = add_InputLayer(row, line_count, templates)
        elif layer_type == 'Conv2D':
            outputs_dict[layer_name], result = add_Conv2D(row, line_count, outputs_dict, templates)
        elif layer_type == 'MaxPooling2D':
            outputs_dict[layer_name], result = add_MaxPooling2D(row, line_count, outputs_dict, templates)
        elif layer_type == 'BatchNormalization':
            outputs_dict[layer_name], result = add_BatchNormalization(row, line_count, outputs_dict, templates, skip)
        elif layer_type == 'Activation':
            outputs_dict[layer_name], result = add_Activation(row, line_count, outputs_dict, templates)
        elif layer_type == 'AveragePooling2D':
            outputs_dict[layer_name], result = add_AveragePooling2D(row, line_count, outputs_dict, templates)
        elif layer_type == 'ZeroPadding2D':
            outputs_dict[layer_name], result = add_ZeroPadding2D(row, line_count, outputs_dict, templates)
        elif layer_type == 'Flatten':
            outputs_dict[layer_name], result = add_Flatten(row, line_count, outputs_dict, templates)
        elif layer_type == 'Dense':
            outputs_dict[layer_name], result = add_Dense(row, line_count, outputs_dict, templates)
        elif layer_type == 'Dropout':
            outputs_dict[layer_name] = outputs_dict[row['connected_to']]
        elif layer_type == 'Add':
            output, result = add_Add(row, line_count, outputs_dict, templates)
        else:
            print('Undefined Layer: {}'.format(layer_type))

        #Update
        SW_def_func += result[0]
        Shared_static_v += result[1]
        SW_static_v += result[2]
        initialization += result[3]
        SW_functions += result[4]
        print_result += result[5]


    # Generate C file
    f = {'Bin_type':dtype,
         'def_SW_functions':SW_def_func,
         'Shared_static_variables':Shared_static_v,
         'SW_static_variables':SW_static_v,
         'Initialization':initialization,
         'SW_functions':SW_functions,
         'result':print_result}
    c_file = templates['main'].substitute(f) + "\n\t";
    file = open('C_Verifier.cpp','w')
    file.write(c_file)
    file.close()
