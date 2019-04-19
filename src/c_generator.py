from .model import *
import sys
from string import Template


class c_generator(object):

    def __init__(self, models, paths=[], dtype='DATA_T'):
        self.model_sw = models
        self.model_name = models.model_name
        self.dtype = dtype
        self.output_path = paths[0]
        self.template_path = paths[1]

    def generate(self):
        o1 = open(self.template_path+"/main/main_sw.txt")
        output_path_1 = "\"" + self.output_path + "/output_value/c_output.txt\""
        output_path_2 = "\"" + self.output_path + "/output_value/c_output_num.txt\""
        file = open(self.output_path+'/c_verifier.cpp', 'w')

        sw_def_layer = ''
        initialization = ''
        sw_static_variables = ''
        sw_output_variables = ''
        sw_call_layer = ''
        print_result = ''
        for layer in self.model_sw.layers:
            sw_def_layer += layer.function['code']
            initialization += layer.function['init']
            sw_static_variables += layer.function['static_w']
            sw_output_variables += layer.function['static_o']
            layer.print_result()
            print_result += layer.function['print_result']

        file.write(o1.read().format(sw_def_layer=sw_def_layer,
                                    sw_static_variables=sw_static_variables,
                                    sw_output_variables=sw_output_variables,
                                    Initialization=initialization,
                                    sw_call_layer=self.gen_sw_call_layer(),
                                    result=print_result,
                                    output_path_1 = output_path_1,
                                    output_path_2 = output_path_2
                                    ))
        file.close()
        return self.output_path+'/c_verifier.cpp';

    def gen_sw_call_layer(self):
        sw_call_layer=''
        for layer in self.model_sw.layers :
            layer_type = layer.config['layer_type']
            l_n = layer.layer_odr
            if layer_type == 'Conv2D':
                sw_call_layer += 'printf(\"[c_verifier.cpp]Calculate Conv2D{}\\n\");\n\t'.format(layer.layer_odr)
                inp = self.model_sw.graphs[layer.config['name']]['in'][0]
                if layer.config['use_bias'] == 'True':
                    sw_call_layer += 'SW_{}(O{}_SW,O{line_num}_SW,W{line_num},B{line_num});\n\t'.format(layer.config['name'], inp, line_num=l_n)
                else:
                    sw_call_layer += 'SW_{}(O{}_SW,O{line_num}_SW,W{line_num});\n\t'.format(layer.config['name'], inp,
                                                                                            line_num=l_n)
            elif layer_type == 'SeparableConv2D':
                sw_call_layer += 'printf(\"[c_verifier.cpp]Calculate SeparableConv2D{}\\n\");\n\t'.format(layer.layer_odr)
                inp = self.model_sw.graphs[layer.config['name']]['in'][0]
                sw_call_layer += 'SW_{}(O{}_SW,O{line_num}_SW,W{line_num}_1,W{line_num}_2);\n\t'.format(layer.config['name']
                                                                                                    , inp, line_num=l_n)
            elif layer_type == 'DepthwiseConv2D':
                sw_call_layer += 'printf(\"[c_verifier.cpp]Calculate DepthwiseConv2D{}\\n\");\n\t'.format(l_n)
                inp = self.model_sw.graphs[layer.config['name']]['in'][0]
                sw_call_layer += 'SW_{}(O{}_SW,O{line_num}_SW,W{line_num});\n\t'.format(layer.config['name'], inp,
                                                                                        line_num=l_n)
            elif layer_type == 'ReLU':
                sw_call_layer += 'printf(\"[c_verifier.cpp]Calculate Relu{}\\n\");\n\t'.format(l_n)
                inp = self.model_sw.graphs[layer.config['name']]['in'][0]
                sw_call_layer += 'SW_{}(O{}_SW,O{}_SW);\n\t'.format(layer.config['name'], inp, l_n)

            elif layer_type == 'Cropping2D':
                sw_call_layer += 'printf(\"[c_verifier.cpp]Calculate Cropping2D{}\\n\");\n\t'.format(l_n)
                inp = self.model_sw.graphs[layer.config['name']]['in'][0]
                sw_call_layer += 'SW_{}(O{}_SW,O{}_SW);\n\t'.format(layer.config['name'], inp, l_n)

            elif layer_type == 'BatchNormalization':
                sw_call_layer += 'printf(\"[c_verifier.cpp]Calculate BatchNormalization{}\\n\");\n\t'.format(l_n)
                inp = self.model_sw.graphs[layer.config['name']]['in'][0]
                sw_call_layer += 'SW_{}(O{}_SW,O{}_SW, W{});\n\t'.format(layer.config['name'], inp, l_n, l_n)

            elif layer_type == 'Activation':
                sw_call_layer += 'printf(\"[c_verifier.cpp]Calculate Activation(Relu){}\\n\");\n\t'.format(l_n)
                inp = self.model_sw.graphs[layer.config['name']]['in'][0]
                sw_call_layer += 'SW_{}(O{}_SW,O{}_SW);\n\t'.format(layer.config['name'], inp, l_n)

            elif layer_type == 'MaxPooling2D':
                sw_call_layer += 'printf(\"[c_verifier.cpp]Calculate MaxPooling2D{}\\n\");\n\t'.format(l_n)
                inp = self.model_sw.graphs[layer.config['name']]['in'][0]
                sw_call_layer += 'SW_{}(O{}_SW,O{}_SW);\n\t'.format(layer.config['name'], inp, l_n)

            elif layer_type == 'AveragePooling2D':
                sw_call_layer += 'printf(\"[c_verifier.cpp]Calculate AveragePooling2D{}\\n\");\n\t'.format(l_n)
                inp = self.model_sw.graphs[layer.config['name']]['in'][0]
                sw_call_layer += 'SW_{}(O{}_SW,O{}_SW);\n\t'.format(layer.config['name'], inp, l_n)

            elif layer_type == 'Add':
                sw_call_layer += 'printf(\"[c_verifier.cpp]Calculate Add{}\\n\");\n\t'.format(l_n)
                a1 = self.model_sw.graphs[layer.config['name']]['in'][0]
                a2 = self.model_sw.graphs[layer.config['name']]['in'][1]
                sw_call_layer += 'SW_{}(O{}_SW,O{}_SW,O{}_SW);\n\t'.format(layer.config['name'],a1,a2,l_n)

            elif layer_type == 'ZeroPadding2D':
                sw_call_layer += 'printf(\"[c_verifier.cpp]Calculate ZeroPadding2D{}\\n\");\n\t'.format(l_n)
                inp = self.model_sw.graphs[layer.config['name']]['in'][0]
                sw_call_layer += 'SW_{}(O{}_SW,O{}_SW);\n\t'.format(layer.config['name'], inp, l_n)

            elif layer_type == 'InputLayer':
                sw_call_layer += 'printf(\"[c_verifier.cpp]InputLayer\\n\");\n\t'

            elif layer_type == 'Flatten':
                sw_call_layer += 'printf(\"[c_verifier.cpp]Calculate Flatten{}\\n\");\n\t'.format(l_n)
                inp = self.model_sw.graphs[layer.config['name']]['in'][0]
                sw_call_layer += 'SW_{}(O{}_SW,O{}_SW);\n\t'.format(layer.config['name'], inp, l_n)

            elif layer_type == 'Dense':
                sw_call_layer += 'printf(\"[c_verifier.cpp]Calculate Dense{}\\n\");\n\t'.format(l_n)
                inp = self.model_sw.graphs[layer.config['name']]['in'][0]
                if layer.config['use_bias'] == 'True':
                    sw_call_layer += 'SW_{}(O{}_SW,O{}_SW,W{},B{});\n\t'.format(layer.config['name'], inp, l_n, l_n, l_n)
                else:
                    sw_call_layer += 'SW_{}(O{}_SW,O{}_SW,W{});\n\t'.format(layer.config['name'], inp, l_n, l_n)

            elif layer_type == 'GlobalAveragePooling2D':
                sw_call_layer += 'printf(\"[c_verifier.cpp]Calculate GlobalAveragePooling2D{}\\n\");\n\t'.format(l_n)
                inp = self.model_sw.graphs[layer.config['name']]['in'][0]
                sw_call_layer += 'SW_{}(O{}_SW,O{}_SW);\n\t'.format(layer.config['name'], inp, l_n)

            elif layer_type == 'GlobalMaxPooling2D':
                sw_call_layer += 'printf(\"[c_verifier.cpp]Calculate GlobalMaxPooling2D{}\\n\");\n\t'.format(l_n)
                inp = self.model_sw.graphs[layer.config['name']]['in'][0]
                sw_call_layer += 'SW_{}(O{}_SW,O{}_SW);\n\t'.format(layer.config['name'], inp, l_n)

            elif layer_type == 'Concatenate':
                sw_call_layer += 'printf(\"[c_verifier.cpp]Calculate Concatenate{}\\n\");\n\t'.format(l_n)
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
                sw_call_layer += 'printf(\"[c_verifier.cpp]Calculate Lambda{}\\n\");\n\t'.format(l_n)
                if len(self.model_sw.graphs[layer.config['name']]['in']) == 2:
                    inp1 = self.model_sw.graphs[layer.config['name']]['in'][0]
                    inp2 = self.model_sw.graphs[layer.config['name']]['in'][1]
                    sw_call_layer += 'SW_{}(O{}_SW, O{}_SW, O{}_SW);\n\t'.format(layer.config['name'], inp1, inp2, l_n)

            elif layer_type == 'Dropout':
                sw_call_layer += 'printf(\"[c_verifier.cpp]Calculate Dropout{}\\n\");\n\t'.format(l_n)
                inp = self.model_sw.graphs[layer.config['name']]['in'][0]
                sw_call_layer += 'SW_{}(O{}_SW,O{}_SW);\n\t'.format(layer.config['name'], inp, l_n)

            elif layer_type == 'Reshape':
                sw_call_layer += 'printf(\"[c_verifier.cpp]Calculate Reshape{}\\n\");\n\t'.format(l_n)
                inp = self.model_sw.graphs[layer.config['name']]['in'][0]
                sw_call_layer += 'SW_{}(O{}_SW,O{}_SW);\n\t'.format(layer.config['name'], inp, l_n)

        return sw_call_layer