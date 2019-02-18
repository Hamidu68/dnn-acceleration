from ..Models import *
from ..Layers import *
import csv
import sys
from string import Template


class CodeGenerators(object):

    def __init__(self, models=[], dtype='DATA_T',name=''):
        self.models = []
        self.dtype = dtype

    def gen_print_result(self):
        print_result = ''
        for layer in self.model_sw.layers :
            layer_type=layer.config['layer_type']
            output_shape = eval(layer.config['batch_output_shape'])
            o1 = open("Code_Generator/Template/Print/Print_Output3D.txt")
            o2 = open("Code_Generator/Template/Print/Print_Output1D.txt")
            output3d = o1.read()
            output1d = o2.read()
            if len(output_shape) <= 2:
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
                i_input = open("Code_Generator/Template/Init/Input_var_Initializer_f.txt")
                init_input = i_input.read()
                func = init_input.format(Input_channel=input_shape[3], Input_width=input_shape[1],
                                         Input_height=input_shape[2])
                initialization += func + "\n\t"
            elif layer_type == 'Conv2D':
                filter_shape = eval(layer.config['kernel_size'])
                c_input = open("Code_Generator/Template/Init/Conv_var_Initializer_f.txt")
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
            elif layer_type == 'Dense':
                d_input = open("Code_Generator/Template/Init/Dense_var_Initializer_f.txt")
                dense_input = d_input.read()
                begin = ''
                end = ''
                if layer.config['use_bias'] == 'False':
                    begin = '/*'
                    end = '*/'
                func = dense_input.format(Input_channel=input_shape[1], Output_channel=output_shape[1],
                                          line_number=l_n, comment_begin=begin, comment_end=end)
                initialization += func + "\n\t"
            elif layer_type == 'BatchNormalization':
                b_input = open("Code_Generator/Template/Init/Batch_var_Initializer_f.txt")
                batch_input = b_input.read()
                if layer.config['scale'] == 'False':
                    func = batch_input.format(Output_channel=output_shape[3], line_number=l_n, num=3)
                else:
                    func = batch_input.format(Output_channel=output_shape[3], line_number=l_n, num=4)
                initialization += func + "\n\t"
        return initialization

    def generate(self):
        return

##################################SW Func##########################################

    def gen_sw_def_layer(self):
        sw_def_layer=''
        for layer in self.model_sw.layers:

            layer_type = layer.config['layer_type']

            if layer_type == 'Conv2D':
                sw_def_layer += layer.function['code']
            elif layer_type == 'BatchNormalization':
                sw_def_layer += layer.function['code']
            elif layer_type == 'Activation':
                sw_def_layer += layer.function['code']
            elif layer_type == 'MaxPooling2D':
                sw_def_layer += layer.function['code']
            elif layer_type == 'AveragePooling2D':
                sw_def_layer += layer.function['code']
            elif layer_type == 'Lambda':
                sw_def_layer += layer.function['code']
            elif layer_type == 'Dense':
                sw_def_layer += layer.function['code']
            elif layer_type == 'Flatten' :
                sw_def_layer += layer.function['code']
            elif layer_type == 'Concatenate':
                sw_def_layer += layer.function['code']
            elif layer_type == 'GlobalAveragePooling2D':
                sw_def_layer += layer.function['code']
            elif layer_type == 'SeparableConv2D':
                sw_def_layer += layer.function['code']
            elif layer_type == 'Cropping2D':
                sw_def_layer += layer.function['code']
            elif layer_type == 'GlobalMaxPooling2D':
                sw_def_layer += layer.function['code']
            elif layer_type == 'ReLU':
                sw_def_layer += layer.function['code']
            elif layer_type == 'DepthwiseConv2D':
                sw_def_layer += layer.function['code']
            elif layer_type == 'Dropout':
                sw_def_layer += layer.function['code']
            elif layer_type == 'Reshape':
                sw_def_layer += layer.function['code']
        return sw_def_layer

    def gen_sw_static_variables(self):
        sw_static_variables = ''
        for layer in self.model_sw.layers :
            output_shape = eval(layer.config['batch_output_shape'])
            input_shape = eval(layer.config['batch_input_shape'])
            layer_type = layer.config['layer_type']
            l_n = layer.layer_odr
            if layer_type == 'Conv2D':
                filter_shape = eval(layer.config['kernel_size'])
                sw_static_variables += 'static DATA_T W{}[{}][{}][{}][{}];\n\t'.format(l_n,output_shape[3],input_shape[3], filter_shape[0], filter_shape[1])
                if layer.config['use_bias'] == 'True':
                    sw_static_variables += 'static DATA_T B{}[{}];\n\t'.format(l_n, output_shape[3])
            elif layer_type == 'SeparableConv2D':
                filter_shape = eval(layer.config['kernel_size'])
                if layer.config['use_bias'] == 'False':
                    sw_static_variables += 'static DATA_T W{}_1[{}][{}][{}];\n\t'.format(l_n, input_shape[3], filter_shape[0], filter_shape[1])
                    sw_static_variables += 'static DATA_T W{}_2[{}][{}];\n\t'.format(l_n, output_shape[3], input_shape[3])
            elif layer_type == 'DepthwiseConv2D':
                filter_shape = eval(layer.config['kernel_size'])
                sw_static_variables += 'static DATA_T W{}[{}][{}][{}];\n\t'.format(l_n, output_shape[3], filter_shape[0],filter_shape[1])
            elif layer_type == 'InputLayer':
                sw_static_variables += 'static DATA_T I[{}][{}][{}];\n\t'.format(input_shape[3], input_shape[1], input_shape[2])
            elif layer_type == 'Dense':
                if layer.config['use_bias'] == 'True':
                    sw_static_variables += 'static DATA_T B{}[{}];\n\t'.format(l_n, output_shape[1])
                sw_static_variables += 'static DATA_T W{}[{}][{}];\n\t'.format(l_n, output_shape[1], input_shape[1])
            elif layer_type == 'BatchNormalization':
                if layer.config['scale'] == 'False':
                    sw_static_variables += 'static DATA_T W{}[3][{}];\n\t'.format(l_n, output_shape[3])
                else:
                    sw_static_variables += 'static DATA_T W{}[4][{}];\n\t'.format(l_n, output_shape[3])
        return sw_static_variables

    def gen_sw_output_variables(self):
        sw_output_variables=''
        for layer in self.model_sw.layers :
            output_shape = eval(layer.config['batch_output_shape'])
            layer_type = layer.config['layer_type']
            l_n=layer.layer_odr
            if layer_type == 'Flatten' or layer_type == 'Dense' or layer_type == 'GlobalAveragePooling2D' or layer_type == 'GlobalMaxPooling2D':
                sw_output_variables += 'static DATA_T O{}_SW[{}];\n\t'.format(l_n, output_shape[1])
            else :
                sw_output_variables += 'static DATA_T O{}_SW[{}][{}][{}];\n\t'.format(l_n, output_shape[3],
                                                                                      output_shape[1], output_shape[2])
        return sw_output_variables

    def gen_sw_call_layer(self):
        sw_call_layer=''
        for layer in self.model_sw.layers :
            layer_type = layer.config['layer_type']
            l_n = layer.layer_odr
            if layer_type == 'Conv2D':
                sw_call_layer += 'printf(\"[C_verifier.cpp]Calculate Conv2D{}\\n\\n\");\n\t'.format(layer.layer_odr)
                inp = self.model_sw.graphs[layer.config['name']]['in'][0]
                if layer.config['use_bias'] == 'True':
                    sw_call_layer += 'SW_{}(O{}_SW,O{line_num}_SW,W{line_num},B{line_num});\n\t'.format(layer.config['name']
                                                                                                    , inp, line_num=l_n)
                else:
                    sw_call_layer += 'SW_{}(O{}_SW,O{line_num}_SW,W{line_num});\n\t'.format(layer.config['name'], inp,
                                                                                            line_num=l_n)
            elif layer_type == 'SeparableConv2D':
                sw_call_layer += 'printf(\"[C_verifier.cpp]Calculate SeparableConv2D{}\\n\\n\");\n\t'.format(layer.layer_odr)
                inp = self.model_sw.graphs[layer.config['name']]['in'][0]
                sw_call_layer += 'SW_{}(O{}_SW,O{line_num}_SW,W{line_num}_1,W{line_num}_2);\n\t'.format(layer.config['name']
                                                                                                    , inp, line_num=l_n)
            elif layer_type == 'DepthwiseConv2D':
                sw_call_layer += 'printf(\"[C_verifier.cpp]Calculate DepthwiseConv2D{}\\n\\n\");\n\t'.format(l_n)
                inp = self.model_sw.graphs[layer.config['name']]['in'][0]
                sw_call_layer += 'SW_{}(O{}_SW,O{line_num}_SW,W{line_num});\n\t'.format(layer.config['name'], inp,
                                                                                        line_num=l_n)
            elif layer_type == 'ReLU':
                sw_call_layer += 'printf(\"[C_verifier.cpp]Calculate Relu{}\\n\\n\");\n\t'.format(l_n)
                inp = self.model_sw.graphs[layer.config['name']]['in'][0]
                sw_call_layer += 'SW_{}(O{}_SW,O{}_SW);\n\t'.format(layer.config['name'], inp, l_n)

            elif layer_type == 'Cropping2D':
                sw_call_layer += 'printf(\"[C_verifier.cpp]Calculate Cropping2D{}\\n\\n\");\n\t'.format(l_n)
                inp = self.model_sw.graphs[layer.config['name']]['in'][0]
                sw_call_layer += 'SW_{}(O{}_SW,O{}_SW);\n\t'.format(layer.config['name'], inp, l_n)

            elif layer_type == 'BatchNormalization':
                sw_call_layer += 'printf(\"[C_verifier.cpp]Calculate BatchNormalization{}\\n\\n\");\n\t'.format(l_n)
                inp = self.model_sw.graphs[layer.config['name']]['in'][0]
                sw_call_layer += 'SW_{}(O{}_SW,O{}_SW, W{});\n\t'.format(layer.config['name'], inp, l_n, l_n)

            elif layer_type == 'Activation':
                sw_call_layer += 'printf(\"[C_verifier.cpp]Calculate Activation(Relu){}\\n\\n\");\n\t'.format(l_n)
                inp = self.model_sw.graphs[layer.config['name']]['in'][0]
                sw_call_layer += 'SW_{}(O{}_SW,O{}_SW);\n\t'.format(layer.config['name'], inp, l_n)

            elif layer_type == 'MaxPooling2D':
                sw_call_layer += 'printf(\"[C_verifier.cpp]Calculate MaxPooling2D{}\\n\\n\");\n\t'.format(l_n)
                inp = self.model_sw.graphs[layer.config['name']]['in'][0]
                sw_call_layer += 'SW_{}(O{}_SW,O{}_SW);\n\t'.format(layer.config['name'], inp, l_n)

            elif layer_type == 'AveragePooling2D':
                sw_call_layer += 'printf(\"[C_verifier.cpp]Calculate AveragePooling2D{}\\n\\n\");\n\t'.format(l_n)
                inp = self.model_sw.graphs[layer.config['name']]['in'][0]
                sw_call_layer += 'SW_{}(O{}_SW,O{}_SW);\n\t'.format(layer.config['name'], inp, l_n)

            elif layer_type == 'Add':
                sw_call_layer += 'printf(\"[C_verifier.cpp]Calculate Add{}\\n\\n\");\n\t'.format(l_n)
                a1 = self.model_sw.graphs[layer.config['name']]['in'][0]
                a2 = self.model_sw.graphs[layer.config['name']]['in'][1]
                sw_call_layer += 'SW_{}(O{}_SW,O{}_SW,O{}_SW);\n\t'.format(layer.config['name'],a1,a2,l_n)

            elif layer_type == 'ZeroPadding2D':
                sw_call_layer += 'printf(\"[C_verifier.cpp]Calculate ZeroPadding2D{}\\n\\n\");\n\t'.format(l_n)
                inp = self.model_sw.graphs[layer.config['name']]['in'][0]
                sw_call_layer += 'SW_{}(O{}_SW,O{}_SW);\n\t'.format(layer.config['name'], inp, l_n)

            elif layer_type == 'InputLayer':
                sw_call_layer += 'printf(\"[C_verifier.cpp]InputLayer\\n\\n\");\n\t'

            elif layer_type == 'Flatten':
                sw_call_layer += 'printf(\"[C_verifier.py]Calculate Flatten{}\\n\\n\");\n\t'.format(l_n)
                inp = self.model_sw.graphs[layer.config['name']]['in'][0]
                sw_call_layer += 'SW_{}(O{}_SW,O{}_SW);\n\t'.format(layer.config['name'], inp, l_n)

            elif layer_type == 'Dense':
                sw_call_layer += 'printf(\"[C_verifier.cpp]Calculate Dense{}\\n\\n\");\n\t'.format(l_n)
                inp = self.model_sw.graphs[layer.config['name']]['in'][0]
                if layer.config['use_bias'] == 'True':
                    sw_call_layer += 'SW_{}(O{}_SW,O{}_SW,W{},B{});\n\t'.format(layer.config['name'], inp, l_n, l_n, l_n)
                else:
                    sw_call_layer += 'SW_{}(O{}_SW,O{}_SW,W{});\n\t'.format(layer.config['name'], inp, l_n, l_n)

            elif layer_type == 'GlobalAveragePooling2D':
                sw_call_layer += 'printf(\"[C_verifier.cpp]Calculate GlobalAveragePooling2D{}\\n\\n\");\n\t'.format(l_n)
                inp = self.model_sw.graphs[layer.config['name']]['in'][0]
                sw_call_layer += 'SW_{}(O{}_SW,O{}_SW);\n\t'.format(layer.config['name'], inp, l_n)

            elif layer_type == 'GlobalMaxPooling2D':
                sw_call_layer += 'printf(\"[C_verifier.cpp]Calculate GlobalMaxPooling2D{}\\n\\n\");\n\t'.format(l_n)
                inp = self.model_sw.graphs[layer.config['name']]['in'][0]
                sw_call_layer += 'SW_{}(O{}_SW,O{}_SW);\n\t'.format(layer.config['name'], inp, l_n)

            elif layer_type == 'Concatenate':
                sw_call_layer += 'printf(\"[C_verifier.cpp]Calculate Concatenate{}\\n\\n\");\n\t'.format(l_n)
                if len(self.model_sw.graphs[layer.config['name']]['in']) == 2:
                    inp1 = self.model_sw.graphs[layer.config['name']]['in'][0]
                    inp2 = self.model_sw.graphs[layer.config['name']]['in'][1]
                    sw_call_layer += 'SW_{}(O{}_SW, O{}_SW, O{}_SW);\n\t'.format(layer.config['name'], inp1, inp2, l_n)
                elif len(self.model_sw.graphs[layer.config['name']]['in']) == 3:
                    inp1 = self.model_sw.graphs[layer.config['name']]['in'][0]
                    inp2 = self.model_sw.graphs[layer.config['name']]['in'][1]
                    inp3 = self.model_sw.graphs[layer.config['name']]['in'][2]
                    sw_call_layer += 'SW_{}(O{}_SW, O{}_SW, O{}_SW,O{}_SW);\n\t'.format(layer.config['name'],
                                                                                        inp1, inp2, inp3, l_n)
                elif len(self.model_sw.graphs[layer.config['name']]['in']) == 4:
                    inp1 = self.model_sw.graphs[layer.config['name']]['in'][0]
                    inp2 = self.model_sw.graphs[layer.config['name']]['in'][1]
                    inp3 = self.model_sw.graphs[layer.config['name']]['in'][2]
                    inp4 = self.model_sw.graphs[layer.config['name']]['in'][3]
                    sw_call_layer += 'SW_{}(O{}_SW, O{}_SW, O{}_SW, O{}_SW,O{}_SW);\n\t'.format(layer.config['name'],
                                                                                                inp1, inp2, inp3, inp4, l_n)
            elif layer_type == 'Lambda':
                sw_call_layer += 'printf(\"[C_verifier.cpp]Calculate Lambda{}\\n\\n\");\n\t'.format(l_n)
                if len(self.model_sw.graphs[layer.config['name']]['in']) == 2:
                    inp1 = self.model_sw.graphs[layer.config['name']]['in'][0]
                    inp2 = self.model_sw.graphs[layer.config['name']]['in'][1]
                    sw_call_layer += 'SW_{}(O{}_SW, O{}_SW, O{}_SW);\n\t'.format(layer.config['name'], inp1, inp2, l_n)

            elif layer_type == 'Dropout':
                sw_call_layer += 'printf(\"[C_verifier.py]Calculate Dropout{}\\n\\n\");\n\t'.format(l_n)
                inp = self.model_sw.graphs[layer.config['name']]['in'][0]
                sw_call_layer += 'SW_{}(O{}_SW,O{}_SW);\n\t'.format(layer.config['name'], inp, l_n)

            elif layer_type == 'Reshape':
                sw_call_layer += 'printf(\"[C_verifier.cpp]Calculate Reshape{}\\n\\n\");\n\t'.format(l_n)
                inp = self.model_sw.graphs[layer.config['name']]['in'][0]
                sw_call_layer += 'SW_{}(O{}_SW,O{}_SW);\n\t'.format(layer.config['name'], inp, l_n)

        return sw_call_layer