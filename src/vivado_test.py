import os, sys, csv
from model import Models
from .vivado_generator import *


def vivado_test(model_info='', model_name='', dtype='DATA_T', batch='', paths=[]):

    output_path = paths[0]
    template_path = paths[1]

    # Read Layer Information from CSV
    csv_reader = csv.DictReader(open(model_info))

    # Make model instance
    models = [Models(model_name=model_name, dtype='DATA_T', post='HW'),
     Models(model_name=model_name, dtype='DATA_T', post='SW')]

    # Skip layers
    if batch == "True":
        skip_layers = ['Dropout']
    else:
        skip_layers = ['Dropout', 'BatchNormalization']

    # for each layers
    for row in csv_reader:
        # check skip layer
        if row["layer_type"] in skip_layers:
            models[0].skip_layer(row)
            models[1].skip_layer(row)
        else:
            models[0].add_layer(row)
            models[1].add_layer(row)
    models[0].set_output()
    models[1].set_output()

    # Generate code
    code_gen = vivado_generator(models=models, paths = paths, dtype = dtype)
    code_gen.generate()