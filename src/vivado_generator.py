from .model import *

class vivado_generator(object):
    def __init__(self, models=[], paths=[],dtype='DATA_T'):
        self.model_hw = models[0]
        self.model_sw = models[1]
        self.dtype = dtype;
        self.model_name = models[0].model_name;
        self.output_path = paths[0]
        self.template_path = paths[1]
        self.sw_call_layer = ''
        self.hw_call_layer = ''
        self.stream_def = ''

    def generate(self):
        o0 = open(self.template_path + "main/vivado_cpp_3D.txt")
    	o1 = open(self.template_path + "main/vivado_cpp_1D.txt")
    	o2 = open(self.template_path + "main/vivado_test_cpp_3D.txt")
        o3 = open(self.template_path + "main/vivado_test_cpp_1D.txt")
    	f1 = open(self.output_path +'/'+ self.model_name+'.cpp', 'w')
        f2 = open(self.output_path +'/'+ self.model_name+'_test.cpp', 'w')

        st=open(self.template_path + "function/Stream_io.txt")
        st_d=open(self.template_path + "function/Stream_io_dense.txt")

        sw_def_layer = ''
        hw_def_layer = ''
        initialization = ''
        static_variables = ''
        i_static_variables = ''
        i_func_params = ''
        i_call_parmas = ''
        optimized_code = ''
        call_params =''
        func_params =''
        sw_output_variables = ''
        copy_values = ''

        # get variables we need
        for layer in self.model_sw.layers:
            sw_def_layer += layer.function['code']
            initialization += layer.function['init']
            static_variables += layer.function['static_w']
            func_params += layer.function['func_p']
            call_params += layer.function['call_p']
            if layer!=self.model_sw.layers[0] and layer!=self.model_sw.layers[-1]:
                sw_output_variables += layer.function['static_o']

        for layer in self.model_hw.layers:
            hw_def_layer += layer.function['code']
            i_static_variables += layer.function['static_w']
            i_func_params += layer.function['func_p']
            i_call_parmas += layer.function['call_p']
            optimized_code += layer.function['optimized_code']
            copy_values += layer.function['copy_values']

        self.gen_call_layer()
        start_layer = self.model_sw.layers[0]
        final_layer = self.model_sw.layers[-1]
        layer_odr = final_layer.layer_odr
        start_layer.set_output_name('O0_SW')
        static_variables +=start_layer.function['static_o']
        final_layer.set_output_name('O_SW')
        static_variables += final_layer.function['static_o']
        final_layer.set_output_name('O_HW')
        static_variables += final_layer.function['static_o']

        # Output 1D, 3D
        if len(final_layer.output.shape) == 3:
            #fill template
            stream_io = st.read().format(
            Input_channel=start_layer.output.shape[2],
            Input_width = start_layer.output.shape[0],
            Input_height = start_layer.output.shape[1],
            Output_channel = final_layer.output.shape[2],
            Output_width = final_layer.output.shape[0],
            Output_height = final_layer.output.shape[1])
            copy_output = "\tfor(m=0; m<"+str(final_layer.output.shape[2])+"; m++) {\n\t\tfor(x=0;x<"+str(final_layer.output.shape[0])+";x++){\n\t\t\tfor(y=0;y<"+str(final_layer.output.shape[1])+";y++){ O[m][x][y] = O_i[m][x][y]; \n}}}\n"
            test_cpp = o2.read().format(
            Data_type= self.dtype,
            model_name=self.model_name,
            func_params= func_params,
            static_variables=static_variables,
            Initialization=initialization,
            call_params= call_params,
            output_channel = final_layer.output.shape[2],
            output_width = final_layer.output.shape[0],
            output_height = final_layer.output.shape[1])
            cpp = o0.read().format(Data_type = self.dtype,
            hw_static_variables= i_static_variables,
            Stream_io= stream_io,
            output_channel = final_layer.output.shape[2],
            output_width = final_layer.output.shape[0],
            output_height = final_layer.output.shape[1],
            hw_def_layer=hw_def_layer,
            sw_def_layer=sw_def_layer,
            i_func_params = i_func_params,
            func_params= func_params,
            layer_odr = layer_odr,
            HLS_optimization = optimized_code,
            model_name=self.model_name,
            call_params=i_call_parmas,
            hw_output_streams= self.stream_def,
            hw_call_layer=self.hw_call_layer,
            copy_values = copy_values,
            copy_output = copy_output,
            sw_output_variables= sw_output_variables,
            sw_call_layer=self.sw_call_layer)

        else:
            #fill template
            stream_io = st_d.read().format(
            Input_channel=start_layer.output.shape[2],
            Input_width = start_layer.output.shape[0],
            Input_height = start_layer.output.shape[1],
            Output_channel = final_layer.output.shape[0])
            copy_output = "\tfor(m=0; m<"+str(final_layer.output.shape[0])+"; m++) { O[m] = O_i[m]; }\n"
            test_cpp = o3.read().format(
            Data_type= self.dtype,
            model_name=self.model_name,
            func_params= func_params,
            static_variables=static_variables,
            Initialization=initialization,
            call_params= call_params,
            output_channel = final_layer.output.shape[0])
            cpp = o1.read().format(Data_type = self.dtype,
            hw_static_variables= i_static_variables,
            Stream_io= stream_io,
            output_channel = final_layer.output.shape[0],
            hw_def_layer=hw_def_layer,
            sw_def_layer=sw_def_layer,
            i_func_params = i_func_params,
            func_params= func_params,
            layer_odr = layer_odr,
            HLS_optimization = optimized_code,
            model_name=self.model_name,
            call_params=i_call_parmas,
            hw_output_streams= self.stream_def,
            hw_call_layer=self.hw_call_layer,
            copy_values = copy_values,
            copy_output = copy_output,
            sw_output_variables= sw_output_variables,
            sw_call_layer=self.sw_call_layer)

        f1.write(cpp);
        f2.write(test_cpp)
        f1.close()
        f2.close()


    # write hw call layer and sw call layer
    def gen_call_layer(self):
        self.hw_call_layer='\tStream_input(I_i,O0_strm);\n\t'
        for layer in self.model_sw.layers :
            layer_type = layer.config['layer_type']
            l_n = layer.layer_odr
            # stream declaration
            self.stream_def += "\tstatic hls::stream<DATA_T> O{line_num}_strm(\"O{line_num}_strm\");\n".format(line_num=l_n)
            # skip inputlayer
            if layer_type == 'InputLayer':
                continue
            # input
            inp = self.model_sw.graphs[layer.config['name']]['in'][0]
            # first call layer
            if l_n == 1 :
                self.sw_call_layer += 'SW_{}(I,'.format(layer.config['name'])
            else :
                self.sw_call_layer += 'SW_{}(O{}_SW,'.format(layer.config['name'],inp)

            # set call layer
            if layer_type == 'Conv2D' or layer_type == 'DepthwiseConv2D' or layer_type == 'Dense' or layer_type == 'BatchNormalization':
                if layer.config['use_bias'] == 'True':
                    self.sw_call_layer += 'O{line_num}_SW,W{line_num},B{line_num});\n\t'.format(line_num=l_n)
                    self.hw_call_layer += 'HW_{}(O{}_strm, W{line_num}_i, B{line_num}_i,O{line_num}_strm);\n\t'.format(layer.config['name'], inp, line_num= l_n)
                else:
                    self.sw_call_layer += 'O{line_num}_SW,W{line_num});\n\t'.format(line_num=l_n)
                    self.hw_call_layer += 'HW_{}(O{}_strm, W{line_num}_i, B{line_num},O{line_num}_strm);\n\t'.format(layer.config['name'], inp, line_num= l_n)

            elif layer_type == 'SeparableConv2D':
                self.sw_call_layer += 'O{line_num}_SW,W{line_num}_1,W{line_num}_2);\n\t'.format(line_num=l_n)
                self.hw_call_layer += 'HW_{}(O{}_strm, W{line_num}_1, W{line_num}_2,O{line_num}_strm);\n\t'.format(layer.config['name'], inp, line_num= l_n)

            elif layer_type == 'Add' or layer_type == 'Lambda':
                # input 2
                a2 = self.model_sw.graphs[layer.config['name']]['in'][1]
                self.sw_call_layer += 'O{}_SW,O{}_SW);\n\t'.format(a2,l_n)
                self.hw_call_layer += 'HW_{}(O{}_strm, O{}_strm, O{}_strm);\n\t'.format(layer.config['name'], inp, a2, l_n)

            elif layer_type == 'Concatenate':
                if len(self.model_sw.graphs[layer.config['name']]['in']) == 2:
                    # input 2
                    inp2 = self.model_sw.graphs[layer.config['name']]['in'][1]
                    self.sw_call_layer += 'O{}_SW, O{}_SW);\n\t'.format(inp2, l_n)
                elif len(self.model_sw.graphs[layer.config['name']]['in']) == 3:
                    # input 2
                    inp2 = self.model_sw.graphs[layer.config['name']]['in'][1]
                    # input 3
                    inp3 = self.model_sw.graphs[layer.config['name']]['in'][2]
                    self.sw_call_layer += 'O{}_SW, O{}_SW,O{}_SW);\n\t'.format(inp2, inp3, l_n)
                elif len(self.model_sw.graphs[layer.config['name']]['in']) == 4:
                    # input2
                    inp2 = self.model_sw.graphs[layer.config['name']]['in'][1]
                    # input 3
                    inp3 = self.model_sw.graphs[layer.config['name']]['in'][2]
                    # input 4
                    inp4 = self.model_sw.graphs[layer.config['name']]['in'][3]
                    self.sw_call_layer += 'O{}_SW, O{}_SW, O{}_SW,O{}_SW);\n\t'.format(inp2, inp3, inp4, l_n)

            else :
                self.sw_call_layer += 'O{}_SW);\n\t'.format(l_n)
                self.hw_call_layer += 'HW_{}(O{}_strm ,O{}_strm);\n\t'.format(layer.config['name'], inp, l_n)

        self.hw_call_layer +='Stream_output(O{}_strm,O);\n\t'.format(l_n)