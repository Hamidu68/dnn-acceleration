from ..Models import *
from ..Layers import *
import csv
import sys
from string import Template


class CodeGenerators:

    def __init__(self, models=[], dtype='DATA_T'):
        self.models = []
        self.dtype = dtype

    def gen_print_result(self):
        print_result = ''
        for layer in self.model_sw.layers :
            layer_type=layer.config['layer_type']
            output_shape = eval(layer.config['batch_output_shape'])
            o1 = open("cpp_generator/sqeezenet/Template/Print/Print_Output3D.txt")
            o2 = open("cpp_generator/sqeezenet/Template/Print/Print_Output1D.txt")
            output3d = o1.read()
            output1d = o2.read()
            if layer_type == 'GlobalAveragePooling2D' or layer_type == 'GlobalMaxPooling2D':
                func = output1d.format(Name=layer_type, first=output_shape[1], output='O'+str(layer.layer_odr)+'_SW')
                print_result += func+"\n"
            else:
                func = output3d.format(Name=layer_type, third=output_shape[3], second=output_shape[2],
                                       first=output_shape[1], output='O'+str(layer.layer_odr)+'_SW')
                print_result += func+"\n"
        return print_result

    def gen_initialization(self):
        initialization = ''
        for layer in self.model_sw.layers:
            layer_type = layer.config['layer_type']
            l_n = layer.layer_odr
            input_shape = eval(layer.config['batch_input_shape'])
            output_shape = eval(layer.config['batch_output_shape'])
            if layer_type == 'InputLayer':
                i_input = open("cpp_generator/sqeezenet/Template/Init/Input_var_Initializer_f.txt")
                init_input = i_input.read()
                func = init_input.format(Input_channel=input_shape[3], Input_width=input_shape[1],
                                         Input_height=input_shape[2])
                initialization += func + "\n\t"
            elif layer_type == 'Conv2D':
                filter_shape = eval(layer.config['kernel_size'])
                c_input = open("cpp_generator/sqeezenet/Template/Init/Conv_var_Initializer_f.txt")
                conv_input = c_input.read()
                begin = ''
                end = ''
                if layer.config['use_bias'] == 'False':
                    begin = '/*'
                    end = '*/'
                func = conv_input.format(Input_channel=input_shape[3], Output_channel=output_shape[3],
                                         Filter_width=filter_shape[0], Filter_height=filter_shape[1], line_number=l_n,
                                         comment_begin=begin, comment_end=end)
                initialization += func + "\n\t"

        return initialization
    
    def generate(self):
        return


class HW_test(CodeGenerators):
    def __init__(self, models=[], dtype='DATA_T'):
        super().__init__(models, dtype)
        self.model_sw = models[0]
        self.model_hw = models[1]

    def generate(self):
        return


class DAC2017_test(CodeGenerators):
    def __init__(self, models=[], dtype='DATA_T'):
        super().__init__(models, dtype)
        self.model_sw = models[0]
        self.model_dac2017 = models[1]

    def generate(self):
        return


