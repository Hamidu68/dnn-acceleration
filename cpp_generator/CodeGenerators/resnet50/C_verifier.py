from ...Models import *
from ...Layers import *
from ..CodeGenerators import *


class C_verifier(CodeGenerators):

    def __init__(self, models=[], dtype='DATA_T'):
        super().__init__(models, dtype)
        self.model_sw = models[0]

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
            elif layer_type == 'Add':
                sw_def_layer += layer.function['code']
            elif layer_type == 'ZeroPadding2D' :
                sw_def_layer += layer.function['code']
            elif layer_type == 'Flatten' :
                sw_def_layer += layer.function['code']
            elif layer_type == 'Dense' :
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
                sw_static_variables += 'static DATA_T B{}[{}];\n\t'.format(l_n, output_shape[3])
            elif layer_type == 'InputLayer':
                sw_static_variables += 'static DATA_T I[{}][{}][{}];\n\t'.format(input_shape[3], input_shape[1], input_shape[2])
            elif layer_type == 'Dense':
                sw_static_variables += 'static DATA_T B{}[{}];\n\t'.format(l_n, output_shape[1])
                sw_static_variables += 'static DATA_T W{}[{}][{}];\n\t'.format(l_n, output_shape[1], input_shape[1])
            elif layer_type == 'BatchNormalization':
                sw_static_variables += 'static DATA_T W{}[4][{}];\n\t'.format(l_n, output_shape[3])
        return sw_static_variables

    def gen_sw_output_variables(self):
        sw_output_variables=''
        for layer in self.model_sw.layers :
            output_shape = eval(layer.config['batch_output_shape'])
            layer_type = layer.config['layer_type']
            l_n=layer.layer_odr
            if layer_type == 'Flatten' or layer_type == 'Dense':
                sw_output_variables += 'static DATA_T O{}_SW[{}];\n\t'.format(l_n,output_shape[1])
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
                sw_call_layer += 'SW_{}(O{}_SW,O{line_num}_SW,W{line_num},B{line_num});\n\t'.format(layer.config['name']
                                                                                                    , inp, line_num=l_n)
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
            elif layer_type == 'Dense' :
                sw_call_layer += 'printf(\"[C_verifier.cpp]Calculate Dense{}\\n\\n\");\n\t'.format(l_n)
                inp = self.model_sw.graphs[layer.config['name']]['in'][0]
                sw_call_layer += 'SW_{}(O{}_SW,O{}_SW,W{},B{});\n\t'.format(layer.config['name'], inp, l_n, l_n, l_n)
        return sw_call_layer

    def generate(self):
        file = open('C_verifier_code/resnet50/C_verifier.cpp', 'w')
        file.write(C_verifier.template.format(sw_def_layer=self.gen_sw_def_layer(),
                                              sw_static_variables=self.gen_sw_static_variables(),
                                              sw_output_variables=self.gen_sw_output_variables(),
                                              Initialization=self.gen_initialization(),
                                              sw_call_layer=self.gen_sw_call_layer(),
                                              result=self.gen_print_result()
                                              ))
        file.close()

    template = """#include <stdio.h>
#include <iostream>
#include <stdlib.h>
#include <string>
#include <string.h>
#include <math.h>
    
using namespace std;

typedef float DATA_T;

{sw_def_layer}

//argv[1] = init_weight.txt, argv[3] = init_input.txt
int main(int argc, char *argv[]){{

    DATA_T temp;
    int m, x, y, i, j, k, l;
    int trash;

    {sw_static_variables}

    {sw_output_variables}

    FILE *w_stream = fopen(argv[1], "rb");
    if (w_stream == NULL) printf("weight file was not opened");
    FILE *i_stream = fopen(argv[2], "rb");
    if (i_stream == NULL) printf("input file was not opened");
    FILE *o_stream = fopen("../../cpp_generator/Output/resnet50/C_output.txt", "w");
    if (o_stream == NULL) printf("Output file was not opened");
    FILE *c_num = fopen("../../cpp_generator/Output/resnet50/c_output_num.txt", "w");
    if (c_num == NULL) printf("Output file was not opened");

    printf("[C_verifier.cpp]Start Initialzation");
    {Initialization}
    printf("[C_verifier.cpp]Finish Initialization");

    {sw_call_layer}

    printf("[C_verifier.cpp]Print Result");


    {result}


    fclose(w_stream);
    fclose(i_stream);
    fclose(o_stream);
    fclose(c_num);

    return 0;
}}
"""


