import os, sys

from cpp_generator import *

# sys.argv[1] : make_C_verifier (T/F)
# sys.argv[2] : make_HW_test (T/F)
# sys.argv[3] : make_DAC2017_test (T/F)
# sys.argv[4] : test file (./test.csv)
# sys.argv[5] : model_name
# sys.argv[6] : data type (int, unsinged int, float, ap_uint<16>)

if __name__ == '__main__':

    make_C_verifier = eval(sys.argv[1])
    make_HW_test = eval(sys.argv[2])
    make_DAC2017_test = eval(sys.argv[3])
    test_file = sys.argv[4]
    model_name = sys.argv[5]
    dtype = sys.argv[6]

    if make_C_verifier:
        gen_c_verifier(test_file, model_name, dtype)
    if make_HW_test:
        gen_HW_test(test_file, model_name, dtype)
    if make_DAC2017_test:
        gen_DAC2017_test(test_file, model_name, dtype)
