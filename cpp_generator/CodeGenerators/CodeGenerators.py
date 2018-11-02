from ..Models import *
from ..Layers import *
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
            o1 = open("../Template/Print/Print_Output3D.txt")
            o2 = open("../Template/Print/Print_Output1D.txt")
            output3d = Template(o1.read())
            output1d = Template(o2.read())
            if layer_type == 'Flatten' or layer_type == 'Dense':
                c = {'Name': layer_type, 'Output_channel': output_shape[1], 'line_number': line_count}
                print_result += output1d.substitute(c)+"\n"
            else:
                c = {'Name': layer_type,'Output_channel': output_shape[3], 'Output_width': output_shape[1], 'Output_height': output_shape[2], 'line_number': layer.layer_odr}
                print_result += output3d.substitute(c)+"\n"
        return print_result

    def gen_initialization(self):
        initialization = ''
        for layer in self.model_sw.layers:
            layer_type = layer.config['layer_type']
            l_n=layer.layer_odr
            input_shape = eval(layer.config['batch_input_shape'])
            output_shape = eval(layer.config['batch_output_shape'])
            if layer_type == 'InputLayer':
                i_input = open("../Template/Init/Input_var_Initializer_f.txt")
                init_input = Template(i_input.read())
                m = {'Input_channel': input_shape[3], 'Input_width': input_shape[1], 'Input_height': input_shape[2]}
                initialization += init_input.substitute(m) + "\n\t"
            elif layer_type == 'Conv2D':
                filter_shape = eval(layer.config['kernel_size'])
                i_input = open("../Template/Init/Input_var_Initializer_f.txt")
                init_input = Template(i_input.read())
                m = {'Input_channel': input_shape[3], 'Output_channel': output_shape[3], 'Filter_width': filter_shape[0], 'Filter_height': filter_shape[1], 'line_number': l_n}
                initialization += init_input.substitute(m) + "\n\t"
            elif layer_type == 'Dense':
                i_input = open("../Template/Init/Input_var_Initializer_f.txt")
                init_input = Template(i_input.read())
                m = {'Input_channel': input_shape[1], 'Output_channel': output_shape[1], 'line_number': l_n}
                initialization += init_input.substitute(m) + "\n\t"
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


