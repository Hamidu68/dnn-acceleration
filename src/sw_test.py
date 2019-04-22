import os, sys, csv
from model import Models
from .c_generator import *
from .keras_generator import *
from .maximum_error import check_maximum_error


def sw_test(model_info='', model_name='', dtype='DATA_T', batch='', paths=[]):

    output_path = paths[0]
    template_path = paths[1]
    weight_file = paths[2]
    input_file = paths[3]

    # Read Layer Information from CSV
    csv_reader = csv.DictReader(open(model_info))

    # Make model instance
    model = Models(model_name=model_name, dtype='DATA_T', post='SW')
    k_model = Models(model_name=model_name, dtype='DATA_T', post='Keras')

    # Skip layers
    if batch == "True":
        skip_layers = ['Dropout']
    else:
        skip_layers = ['Dropout', 'BathNormalization']

    # for each layers
    for row in csv_reader:
        # check skip layer
        if row["layer_type"] in skip_layers:
            model.skip_layer(row)
        else:
            model.add_layer(row)
    model.set_output()

    # C code
    # Generate c_code.cpp
    code_gen = c_generator(models=model, paths = paths, dtype = dtype)
    cpp_file_path = code_gen.generate()
    out_file_path = output_path+'/a.out'

    #-std=c++0x %s -o outvariable_path=%s'
    # Compile and Run cpp file
    os.system('g++ %s -o %s' % (cpp_file_path, out_file_path))
    os.system('%s %s %s' % (out_file_path, weight_file, input_file))

    csv_reader = csv.DictReader(open(model_info))

    for row in csv_reader:
        k_model.add_layer(row)
    k_model.set_output()
    
    # Keras
    keras_generator(k_model, model_name, dtype, paths, skip_layers)

    # Compare Keras and C result
    cpp_output_path = output_path+'/output_value/c_output.txt'
    keras_output_path = output_path+'/output_value/keras_output.txt'
    os.system('vimdiff %s %s' % (cpp_output_path, keras_output_path))

    # Check maximum error
    cpp_output_path = output_path+'/output_value/c_output_num.txt'
    keras_output_path = output_path+'/output_value/keras_output_num.txt'
    check_maximum_error(cpp_output_path, keras_output_path)
