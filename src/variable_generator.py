import os, sys
from sys import argv
import csv
import numpy as np

def variable_generator(csv_dir='',weight_dir='',input_dir='',Random_range=8,init_dtype='int'):
    # Check data type of output file
    dtype = np.int32
    dtype2 = "int"
    data_dype = init_dtype
    if data_dype == 'int':
        dtype = np.int32
        dtype2 = "int"
    elif data_dype == 'unsigned int':
        dtype = np.uint32
        dtype2 = "int"
    elif data_dype == 'float':
        dtype = np.float32
        dtype2 = "float"
    elif data_dype == 'ap_uint<16>':
        dtype = np.uint16
        dtype2 = "int"
    else:
        print('Wrong data type!')

    csv_dir = csv_dir
    weight_dir = weight_dir
    input_dir  = input_dir
    random_range = Random_range

    # open csv
    csv_file = open(csv_dir)
    csv_reader = csv.DictReader(csv_file)

    # parameters
    input_size = 0
    total_params = 0

    # Get total number of parameters
    for row in csv_reader:
        total_params += int(row["params"])
        if row["layer_type"] == "InputLayer":
            input_shape = eval(row["batch_input_shape"])
            input_size += input_shape[3]*input_shape[2]*input_shape[1]

    # open File
    Weight = open(weight_dir, 'wb')
    Input  = open(input_dir, 'wb')

    # Make random initialized file
    if dtype2 == "int":
        np.random.randint(low=-1, high=random_range+1, size=total_params, dtype=dtype).tofile(Weight)
        np.random.randint(low=-1, high=random_range+1, size=input_size, dtype=dtype).tofile(Input)

    elif dtype2 == "float":
        np.random.uniform(low=0, high=random_range, size=total_params).astype(dtype).tofile(Weight)
        np.random.uniform(low=0, high=random_range, size=input_size).astype(dtype).tofile(Input)
    # Close file
    Weight.close()
    Input.close()