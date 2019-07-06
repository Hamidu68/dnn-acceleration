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
                                    output_path_2 = output_path_2,
                                    Data_type = self.dtype
                                    ))
        file.close()
        return self.output_path+'/c_verifier.cpp';

    def gen_sw_call_layer(self):
        sw_call_layer=''
        for layer in self.model_sw.layers :
            layer_type = layer.config['layer_type']
            l_n = layer.layer_odr
            if l_n == 0:
                continue

            sw_call_layer += 'printf(\"[c_verifier.cpp]Calculate {}{}\\n\");\n\t'.format(layer_type,layer.layer_odr)
            if l_n == 1:
                sw_call_layer += 'SW_{}(I,'.format(layer.config['name'])
            else :
                inp = self.model_sw.graphs[layer.config['name']]['in'][0]
                sw_call_layer += 'SW_{}(O{}_SW,'.format(layer.config['name'],inp)

            if layer_type in ['Conv2D','Dense']:
                if layer.config['use_bias'] == 'True':
                    sw_call_layer += 'O{line_num}_SW,W{line_num},B{line_num});\n\t'.format(line_num=l_n)
                else:
                    sw_call_layer += 'O{line_num}_SW,W{line_num});\n\t'.format(line_num=l_n)

            elif layer_type == 'SeparableConv2D':
                sw_call_layer += 'O{line_num}_SW,W{line_num}_1,W{line_num}_2);\n\t'.format(line_num=l_n)

            elif layer_type in ['DepthwiseConv2D','BatchNormalization']:
                sw_call_layer += 'O{line_num}_SW,W{line_num});\n\t'.format(line_num=l_n)

            elif layer_type in ['ReLU','Cropping2D','Activation','MaxPooling2D','AveragePooling2D','ZeroPadding2D','Flatten','GlobalAveragePooling2D','GlobalMaxPooling2D','Dropout','Reshape'] :
                sw_call_layer += 'O{}_SW);\n\t'.format(l_n)

            elif layer_type in ['Add','Lambda']:
                inp2 = self.model_sw.graphs[layer.config['name']]['in'][1]
                sw_call_layer += 'O{}_SW,O{}_SW);\n\t'.format(inp2,l_n)

            elif layer_type == 'Concatenate':
                if len(self.model_sw.graphs[layer.config['name']]['in']) == 2:
                    inp2 = self.model_sw.graphs[layer.config['name']]['in'][1]
                    sw_call_layer += 'O{}_SW, O{}_SW);\n\t'.format(inp2, l_n)
                elif len(self.model_sw.graphs[layer.config['name']]['in']) == 3:
                    inp2 = self.model_sw.graphs[layer.config['name']]['in'][1]
                    inp3 = self.model_sw.graphs[layer.config['name']]['in'][2]
                    sw_call_layer += 'O{}_SW, O{}_SW,O{}_SW);\n\t'.format(inp2, inp3, l_n)
                elif len(self.model_sw.graphs[layer.config['name']]['in']) == 4:
                    inp2 = self.model_sw.graphs[layer.config['name']]['in'][1]
                    inp3 = self.model_sw.graphs[layer.config['name']]['in'][2]
                    inp4 = self.model_sw.graphs[layer.config['name']]['in'][3]
                    sw_call_layer += 'O{}_SW, O{}_SW, O{}_SW,O{}_SW);\n\t'.format(inp2, inp3, inp4, l_n)

        return sw_call_layer
