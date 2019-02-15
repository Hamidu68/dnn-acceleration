
import os, sys
from sys import argv
import csv
import numpy as np

# sys.argv[1] = Test.csv (layer file) 
# sys.argv[2] = Init_Weight.bin (Output-weight file)
# sys.argv[3] = Init_Input.bin(Output-weight file)
# sys.argv[4] = Random_range
# sys.argv[5] = Init_weight_data_type
if __name__ == "__main__":
    # Check data type of output file
    dtype = np.int32
    dtype2 = "int"
    data_dype = sys.argv[5]
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




    csv_dir = sys.argv[1]
    weight_dir = sys.argv[2]
    input_dir  = sys.argv[3]
    random_range = int(argv[4])


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
        np.random.randint(low=1, high=random_range+1, size=total_params, dtype=dtype).tofile(Weight)
        np.random.randint(low=1, high=random_range+1, size=input_size, dtype=dtype).tofile(Input)
        
    elif dtype2 == "float":
        np.random.uniform(low=0, high=1, size=total_params).astype(dtype).tofile(Weight)
        np.random.uniform(low=0, high=1, size=input_size).astype(dtype).tofile(Input)
    # Close file
    Weight.close()
    Input.close()

