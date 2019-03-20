import os, sys, csv
from .Model import Models
from .sw_generator import *
from .keras_generator import *

from .maximum_error import check_maximum_error


def gen_sw_test(test_file='', model_name='', dtype='int',weight_file='', Input_file='', batch=''):
    # Main) Read Layer Information from CSV
    csv_reader = csv.DictReader(open(test_file))

    # Main) make model instance
    models = [Models(model_name=model_name, dtype='DATA_T', post='_SW')]
    
    # Main) Generate Function depending on layer_type
    # Skip layers
    if batch : 
        skip_layers = ['Dropout']
    else:
        skip_layers = ['Dropout', 'BathNormalization']

    # for each layers
    for row in csv_reader:
        # check skip layer
        if row["layer_type"] in skip_layers:
            for model in models:
                model.skip_layer(row)
        else:
            for model in models:
                model.add_layer(row)

    for model in models:
        model.set_output()

    # generate SW_code.cpp
    code_gen = SWGenerators(models=models, dtype=dtype,name = model_name)
    cpp_file_path = code_gen.generate()

    # Compare cpp and keras
    out_file_path = 'output/'+model_name+'/a.out'

    #-std=c++0x %s -o outvariable_path=%s'
    # Compile and Run cpp file
    os.system('g++ %s -o %s' % (cpp_file_path, out_file_path))
    os.system('%s %s %s' % (out_file_path, weight_file, Input_file))

    csv_reader = csv.DictReader(open(test_file))
    models1 = [Models(model_name=model_name, dtype='DATA_T', post='_SW')]

    # for each layers
    for row in csv_reader:
        for model in models1:
            model.add_layer(row)

    for model in models1:
        model.set_output()

    # Keras-Verification
    Keras_Verifier(models1[0], model_name, weight_file, Input_file, dtype)

    cpp_output_path = 'output/'+model_name+'/output_value/c_output.txt'
    keras_output_path = 'output/'+model_name+'/output_value/keras_output.txt'
    os.system('vimdiff %s %s' % (cpp_output_path, keras_output_path))
    # Check maximum error
    # from maximum_error import check_maximum_error
    cpp_output_path = 'output/'+model_name+'/output_value/c_output_num.txt'
    keras_output_path = 'output/'+model_name+'/output_value/keras_output_num.txt'
    check_maximum_error(cpp_output_path, keras_output_path)
