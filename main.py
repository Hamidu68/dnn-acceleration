import os, sys

from src import *

# sys.argv[1] : run_gen_sw_test (T/F)
# sys.argv[2] : run_gen_hw_test (T/F)
# sys.argv[3] : run_gen_DAC2017_test (T/F)
# sys.argv[4] : test file (./test.csv)
# sys.argv[5] : model_name
# sys.argv[6] : data type (int, unsinged int, float, ap_uint<16>)

if __name__ == '__main__':
    
    run_gen_sw_test = eval(sys.argv[1])
    run_gen_hw_test = eval(sys.argv[2])

    # run_gen_DAC2017_test = eval(sys.argv[3])
    test_file = "./model_info/" + sys.argv[4]
    model_name = sys.argv[5]
    dtype = sys.argv[6]
    batch = sys.argv[7]
    Random_range=5
    weight_file_path = './src/init_weight.bin'
    input_file_path = './src/init_input.bin'
    Use_trained_weight=0
    Trained_weight_file=model_name+"_weights.bin"
    Image_file="image.bin"
    if run_gen_sw_test and Use_trained_weight == 1:
        gen_sw_test(test_file, model_name, dtype,Trained_weight_file,Image_file)
    
    if run_gen_sw_test and Use_trained_weight == 0:
        variable_generator.Variable_Generator(test_file,weight_file_path,input_file_path,Random_range,dtype)
        gen_sw_test(test_file, model_name, dtype, weight_file_path, input_file_path, batch)
    
    # if run_gen_hw_test:
    #    gen_hw_test(test_file, model_name, dtype)

# if run_gen_DAC2017_test:
#    gen_DAC2017_test(test_file, model_name, dtype)

