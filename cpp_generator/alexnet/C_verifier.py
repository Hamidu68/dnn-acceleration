import os, sys, csv
from .Models import *
from .CodeGenerators import *


def gen_c_verifier(test_file='', model_name='', dtype='int'):
    # Main) Read Layer Information from CSV
    csv_reader = csv.DictReader(open(test_file))

    # Main) make model instance
    models = [Models(model_name=model_name, dtype='DATA_T', post='_SW')]
    
    # Main) Generate Function depending on layer_type
    # Skip layers
    skip_layers = []
    
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

    # generate c_verifier.cpp
    code_gen = C_verifier(models=models, dtype=dtype)
    code_gen.generate()
