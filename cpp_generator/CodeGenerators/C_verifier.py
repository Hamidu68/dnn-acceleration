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
            input_shape = eval(layer.config['batch_input_shape'])
            
            layer_type=layer.config['layer_type']
            if layer_type=='Conv2D' :
                filter_shape = eval(layer.config['kernel_size'])
                stride_shape = eval(layer.config['strides'])
                conv_s=open("../Template/Function/Conv2D_same_relu.txt")
                conv_v = open("../Template/Function/Conv2D_valid.txt")
                Conv2D_same = Template(conv_s.read())
                Conv2D_valid = Template(conv_v.read())
                l = {'Name' : row["name"], 'Input_channel' : input_shape[3], 'Input_width' : input_shape[1],'Stride_width': stride_shape[0],'Stride_height':stride_shape[1],
                    'Input_height' : input_shape[2], 'Output_channel' : output_shape[3],'Filter_width' : filter_shape[0],
                    'Filter_height' : filter_shape[1], 'Output_width' : output_shape[1], 'Output_height' : output_shape[2]}
                if layer.config['padding'] == 'valid' :
                    sw_def_layer += Conv2D_valid.substitute(l) +"\n"
                else :
                    sw_def_layer += Conv2D_same.substitute(l) +"\n"
            elif layer_type=='BatchNormalization' :
                batch_normal = open("../Template/Function/BatchNormalization.txt")
                BatchNormalization = Template(batch_normal.read())
                l = {'Name' : layer.config['name'], 'Input_channel' : input_shape[3], 'Input_width' : input_shape[1], 'Input_height' : input_shape[2], 'Output_channel' : output_shape[3],
                    'Output_width' : output_shape[1], 'Output_height' : output_shape[2]}
                sw_def_layer += BatchNormalization.substitute(l) +"\n"
            elif layer_type=='Activation' :
                rl = open("../Template/Function/Relu.txt")
                Relu = Template(rl.read())
                l = {'Name' : row["name"], 'Input_channel' : input_shape[3], 'Input_width' : input_shape[1], 'Input_height' : input_shape[2], 'Output_channel' : output_shape[3],
                    'Output_width' : output_shape[1], 'Output_height' : output_shape[2]}
                sw_def_layer += Relu.substitute(l) +"\n"
            elif layer_type=='MaxPooling2D' :
                pool_shape = eval(layer.config['pool_size'])
                stride_shape = eval(layer.config['strides'])
                mxp = open("../Template/Function/MaxPooling2D.txt")
                MaxPooling2D = Template(mxp.read())
                l = {'Name' : row["name"], 'Input_channel' : input_shape[3], 'Input_width' : input_shape[1], 'Input_height' : input_shape[2], 'Output_channel' : output_shape[3],
                    'Output_width' : output_shape[1], 'Output_height' : output_shape[2], 'Stride_width' : stride_shape[0], 'Stride_height':stride_shape[1],
                    'Pool_width' : pool_shape[0], 'Pool_height' : pool_shape[1]}
                sw_def_layer += MaxPooling2D.substitute(l) +"\n"
            elif layer_type=='AveragePooling2D' :
                stride_shape = eval(layer.config['strides'])
                pool_shape = eval(layer.config['pool_size'])
                avp = open("../Template/Function/AveragePooling2D.txt")
                AveragePooling2D = Template(avp.read())
                l = {'Name' : row["name"], 'Input_channel' : input_shape[3], 'Input_width' : input_shape[1], 'Input_height' : input_shape[2], 'Output_channel' : output_shape[3],
                    'Output_width' : output_shape[1], 'Output_height' : output_shape[2], 'Stride_width' : stride_shape[0], 'Stride_height':stride_shape[1],
                    'Pool_width' : pool_shape[0], 'Pool_height' : pool_shape[1]}
                sw_def_layer+=AveragePooling2D.substitute(l) +"\n"
            elif layer_type=='Add' :
                ad = open("../Template/Function/Add.txt")
                Add = Template(ad.read())
                l = {'Name' : row['name'], 'Input_channel1' : output_shape[3], 'Input_width1' : output_shape[1], 'Input_height1' : output_shape[2],'Input_channel2' : output_shape[3], 'Input_width2' : output_shape[1], 'Input_height2' : output_shape[2],'Output_channel' : output_shape[3], 'Output_width' : output_shape[1], 'Output_height' : output_shape[2]}
                sw_def_layer += Add.substitute(l) +"\n"
            elif layer_type=='ZeroPadding2D' :
                pool_shape = eval(layer.config['pool_size'])
                zp = open("../Template/Function/ZeroPadding.txt")
                ZeroPadding2D = Template(zp.read())
                l = {'Name' : row["name"], 'Input_channel' : input_shape[3], 'Input_width' : input_shape[1], 'Input_height' : input_shape[2], 'Output_channel' : output_shape[3],
                    'Output_width' : output_shape[1], 'Output_height' : output_shape[2], 'Padding_size': padding[0]}
                sw_def_layer += ZeroPadding2D.substitute(l) +"\n"
            elif layer_type=='Flatten' :
                fla = open("../Template/Function/Flatten.txt")
                Flatten = Template(fla.read())
                l = {'Name':row["name"],'Input_channel':input_shape[3],'Input_width':input_shape[1],'Input_height':input_shape[2],'Output_channel':output_shape[1]}
                sw_def_layer += Flatten.substitute(l) + "\n"
            elif layer_type=='Dense' :
                den_s = open("../Template/Function/Dense_Softmax.txt")
                den_r = open("../Template/Function/Dense_Relu.txt")
                Dense_softmax = Template(den_s.read())
                Dense_relu = Template(den_r.read())
                l = {'Name':row["name"],'Input_channel':input_shape[1],'Output_channel':output_shape[1]}
                if layer.config['activation'] == 'relu' : # Activation = relu
                    sw_def_layer += Dense_relu.substitute(l) + "\n"
                else :  # Activation = softmax
                    sw_def_layer += Dense_softmax.substitute(l) + "\n"
        return sw_def_layer

    def gen_sw_static_variables():
        sw_static_variables=''
        for layer in model_sw.layers :
            output_shape = eval(layer.config['batch_output_shape'])
            input_shape = eval(layer.config['batch_input_shape'])
            filter_shape = eval(layer.config['kernel_size'])
            layer_type=layer.config['layer_type']
            l_n=layer.layer_odr
            if layer_type=='Conv2D' :
                sw_static_variables+='static DATA_T W{}[{}][{}][{}][{}];\n\t'.format(l_n,output_shape[3],input_shape[3],filter_shape[0],filter_shape[1])
                sw_static_variables+='static DATA_T B{}[{}];\n\t'.format(l_n,output_shape[3])
            elif layer_type=='InputLayer :
                sw_static_variables+='static DATA_T I[{}][{}][{}];\n\t'.format(input_shape[3],input_shape[1],input_shape[2])
            elif layer_type=='Dense' :
                sw_static_variables+='static DATA_T B{}[{}];\n\t'.format(l_n,output_shape[1])
                sw_static_variables+='static DATA_T W{}[{}][{}];\n\t'.format(l_n,output_shape[1],input_shape[1])
        return sw_static_variables

    def gen_sw_output_variables():
        sw_output_variables=''
        for layer in model_sw.layers :
            output_shape = eval(layer.config['batch_output_shape'])
            layer_type=layer.config['layer_type']
            l_n=layer.layer_odr
            if layer_type=='Flatten' or 'Dense' :
                sw_output_variables+='static DATA_T O{}_SW[{}];\n\t'.format(l_n,output_shape[1])
            else :
                sw_output_variables+='static DATA_T O{}_SW[{}][{}][{}];\n\t'.format(l_n,output_shape[3],output_shape[1],output_shape[2])
        return sw_output_variables

    def gen_sw_call_layer():
        sw_call_layer=''
        for layer in model_sw.layers :
            output_shape = eval(layer.config['batch_output_shape'])
            layer_type=layer.config['layer_type']
            l_n=layer.layer_odr
            if layer_type=='Conv2D' :
                sw_call_layer+='printf(\"[C_verifier.cpp]Calculate Conv2D{}\\n\\n\");\n\t'.format(layer.layer_odr)
                sw_call_layer+='SW_{}(O{}_SW,O{line_num}_SW,B{line_num},W{line_num});\n\t'.format(layer.config['name'],l_n-1,line_num=l_n)
            elif layer_type=='BatchNormalization' :
                sw_call_layer+='printf(\"[C_verifier.cpp]Calculate BatchNormalization{}\\n\\n\");\n\t'.format(l_n)
                sw_call_layer+='SW_{}(O{}_SW,O{}_SW);\n\t'.format(layer.config['name'],l_n-1,l_n)
            elif layer_type=='Activation' :
                sw_call_layer+='printf(\"[C_verifier.cpp]Calculate Activation(Relu){}\\n\\n\");\n\t'.format(l_n)
                sw_call_layer+='SW_{}(O{}_SW,O{}_SW);\n\t'.format(layer.config['name'],l_n-1,l_n)
            elif layer_type=='MaxPooling2D' :
                sw_call_layer+='printf(\"[C_verifier.cpp]Calculate MaxPooling2D{}\\n\\n\");\n\t'.format(l_n)
                sw_call_layer+='SW_{}(O{}_SW,O{}_SW);\n\t'.format(layer.config['name'],l_n-1,l_n)
            elif layer_type=='AveragePooling2D' :
                sw_call_layer+='printf(\"[C_verifier.cpp]Calculate AveragePooling2D{}\\n\\n\");\n\t'.format(l_n)
                sw_call_layer+='SW_{}(O{}_SW,O{}_SW);\n\t'.format(layer.config['name'],l_n-1,l_n)
            elif layer_type=='Add' :
                sw_call_layer+='printf(\"[C_verifier.cpp]Calculate Add{}\\n\\n\");\n\t'.format(l_n)
                a1=model_sw.graphs[layer.config['name']]['in'][0]
                a2=model_sw.graphs[layer.config['name']]['in'][1]
                sw_call_layer+='SW_{}(O{}_SW,O{}_SW,O{}_SW);\n\t'.format(layer.config['name'],a1,a2,l_n)
            elif layer_type=='ZeroPadding2D' :
                sw_call_layer+='printf(\"[C_verifier.cpp]Calculate ZeroPadding2D{}\\n\\n\");\n\t'.format(l_n)
                sw_call_layer+='SW_{}"(O{}_SW,O{}_SW);\n\t'.format(layer.config['name'],l_n-1,l_n)
            elif layer_type=='InputLayer' :
                sw_call_layer+='printf(\"[C_verifier.cpp]InputLayer\\n\\n\");\n\t'
            elif layer_type=='Flatten' :
                sw_call_layer+='printf(\"[C_verifier.py]Calculate Flatten{}\\n\\n\");\n\t'.format(l_n)
                sw_call_layer+='SW_{}"(O{}_SW,O{}_SW);\n\t'.format(layer.config['name'],l_n-1,l_n)
            elif layer_type=='Dense' :
                sw_call_layer+='printf(\"[C_verifier.cpp]Calculate Dense{}\\n\\n\");\n\t'.format(l_n)
                sw_call_layer+='SW_{}(O{}_SW,W{},B{},O{}_SW);\n\t'.format(layer.config['name'],l_n-1,l_n,l_n,l_n)
        return sw_call_layer
    
    def generate():
        f.write(template.format(sw_def_layer = get_sw_def_layer(),
                                sw_static_variables=gen_sw_static_variables(),
                                sw_output_variables=gen_sw_output_variables(),
                                Initialization=gen_Initialization(),
                                sw_call_layer=gen_sw_call_layer(),
                                result=gen_print_result()
                                ))

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
