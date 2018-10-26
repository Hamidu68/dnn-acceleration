from ..Models import *
from ..Layers import *
from .CodeGenerators import CodeGenerators

class C_verifier(CodeGenerators):
    def __init__(models=[], dtype='DATA_T'):
        super().__init__(models, dtype)
        self.model_sw = models[0]

    def gen_sw_def_layer():
        sw_def_layer=''
        for layer in model_sw.layers :
            output_shape = eval(layer.config['batch_output_shape'])
            layer_type=layer.config['layer_type']
            if layer_type=='Conv2D' :
            elif layer_type=='BatchNormalization' :
            elif layer_type=='Activation' :
            elif layer_type=='MaxPooling2D' :
            elif layer_type=='AveragePooling2D' :
            elif layer_type=='Add' :
            elif layer_type=='ZeroPadding2D' :
            elif layer_type=='InputLayer' :
            elif layer_type=='Flatten' :
            elif layer_type=='Dense' :
        return sw_def_layer

    def gen_sw_static_variables():
        sw_static_variables=''
        for layer in model_sw.layers :
            output_shape = eval(layer.config['batch_output_shape'])
            input_shape = eval(layer.config['batch_input_shape'])
            filter_shape = eval(layer.config['kernel_size'])
            layer_type=layer.config['layer_type']
            if layer_type=='Conv2D' :
                sw_static_variables+='static DATA_T W{}[{}][{}][{}][{}];\n\t'.format(layer.layer_odr,output_shape[3],input_shape[3],filter_shape[0],filter_shape[1])
                sw_static_variables+='static DATA_T B{}[{}];\n\t'.format(layer.layer_odr,output_shape[3])
            elif layer_type=='InputLayer :
                sw_static_variables+='static DATA_T I[{}][{}][{}];\n\t'.format(input_shape[3],input_shape[1],input_shape[2])
            elif layer_type=='Dense' :
                sw_static_variables+='static DATA_T B{}[{}];\n\t'.format(layer.layer_odr,output_shape[1])
                sw_static_variables+='static DATA_T W{}[{}][{}];\n\t'.format(layer.layer_odr,output_shape[1],input_shape[1])
        return sw_static_variables

    def gen_sw_output_variables():
        sw_output_variables=''
        for layer in model_sw.layers :
            output_shape = eval(layer.config['batch_output_shape'])
            layer_type=layer.config['layer_type']
            if layer_type=='Flatten' or 'Dense' :
                sw_output_variables+='static DATA_T O{}_SW[{}];\n\t'.format(layer.layer_odr,output_shape[1])
            else :
                sw_output_variables+='static DATA_T O{}_SW[{}][{}][{}];\n\t'.format(layer.layer_odr,output_shape[3],output_shape[1],output_shape[2])
        return sw_output_variables

    def gen_sw_call_layer():
        sw_call_layer=''
        for layer in model_sw.layers :
            output_shape = eval(layer.config['batch_output_shape'])
            layer_type=layer.config['layer_type']
            if layer_type=='Conv2D' :
                sw_call_layer+='printf(\"[C_verifier.cpp]Calculate Conv2D{}\\n\\n\");\n\t'.format(layer.layer_odr)
                sw_call_layer+='SW_{}(O{}_SW,O{line_num}_SW,B{line_num},W{line_num});\n\t'.format(layer.config['name'],layer.layer_odr-1,line_num=layer.layer_odr)
            elif layer_type=='BatchNormalization' :
                sw_call_layer+='printf(\"[C_verifier.cpp]Calculate BatchNormalization{}\\n\\n\");\n\t'.format(layer.layer_odr)
                sw_call_layer+='SW_{}(O{}_SW,O{}_SW);\n\t'.format(layer.config['name'],layer.layer_odr-1,layer.layerf_odr)
            elif layer_type=='Activation' :
                sw_call_layer+='printf(\"[C_verifier.cpp]Calculate Activation(Relu){}\\n\\n\");\n\t'.format(layer.layer_odr)
                sw_call_layer+='SW_{}(O{}_SW,O{}_SW);\n\t'.format(layer.config['name'],layer.layer_odr-1,layer.layer_odr)
            elif layer_type=='MaxPooling2D' :
                sw_call_layer+='printf(\"[C_verifier.cpp]Calculate MaxPooling2D{}\\n\\n\");\n\t'.format(layer.layer_odr)
                sw_call_layer+='SW_{}(O{}_SW,O{}_SW);\n\t'.format(layer.config['name'],layer.layer_odr-1,layer.layer_odr)
            elif layer_type=='AveragePooling2D' :
                sw_call_layer+='printf(\"[C_verifier.cpp]Calculate AveragePooling2D{}\\n\\n\");\n\t'.format(layer.layer_odr)

            elif layer_type=='Add' :
            elif layer_type=='ZeroPadding2D' :
            elif layer_type=='InputLayer' :
            elif layer_type=='Flatten' :
            elif layer_type=='Dense' :
        return sw_call_layer
    
    def generate():
        f.write(template.format(sw_def_layer = get_sw_def_layer(),
                                sw_static_variables=gen_sw_static_variables(),
                                sw_output_variables=gen_sw_output_variables(),
                                Initialization=
                                sw_call_layer=gen_sw_call_layer(),
                                result=
                                ))
        return

    template = '''
#include <stdio.h>
#include<iostream>
#include <stdlib.h>
#include<string>
#include<string.h>
#include<math.h>
using namespace std;

typedef float DATA_T;

{sw_def_layer}

//argv[1] = init_weight.txt , argv[2] = init_bias.txt , argv[3] = init_input.txt
int main(int argc, char *argv[]){
    
        DATA_T temp;
        int m, x, y, i, j, k, l;
        int trash;
                
        {sw_static_variables}
                    
        {sw_output_variables}
                        
        FILE *w_stream = fopen(argv[1], "rb");
        if (w_stream == NULL) printf("weight file was not opened");
        FILE *b_stream = fopen(argv[2], "rb");
        if (b_stream == NULL) printf("bias file was not opened");
        FILE *i_stream = fopen(argv[3], "rb");
        if (i_stream == NULL) printf("input file was not opened");
        FILE *o_stream = fopen("Output/C_output.txt", "w");
        if (o_stream == NULL) printf("Output file was not opened");
        FILE *c_num = fopen("Output/c_output_num.txt", "w");
        if (c_num == NULL) printf("Output file was not opened");
                        
        printf("[C_verifier.cpp]Start Initialzation\n\n");
        {Initialization}
        printf("[C_verifier.cpp]Finish Initialization\n\n");
                            
        {sw_call_layer}
                                    
        printf("[C_verifier.cpp]Print Result\n");

        {result}
                                        
                                        
        fclose(w_stream);
        fclose(b_stream);
        fclose(i_stream);
        fclose(o_stream);
        fclose(c_num);
                                        
        return 0;
}
'''
