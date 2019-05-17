from .Data import Data
from .Layers import Layers
from string import Template


class Conv2D(Layers):

    def __init__(self, config={}, inputs=[], dtype='DATA_T', layer_odr=0, post='') :

        super(Conv2D, self).__init__(config, inputs, dtype, layer_odr, post)

        # get shape
        input_shape = eval(self.config['batch_input_shape'])
        output_shape = eval(self.config['batch_output_shape'])
        filter_shape = eval(self.config['kernel_size'])
        stride_shape = eval(self.config['strides'])
        dilation_rate = eval(self.config['dilation_rate'])
        padding = str(self.config['padding'])
        self.use_bias = eval(self.config['use_bias'])

        # set_output
        self.set_output()

        # set_weight
        weight_shape=(output_shape[3], input_shape[3], filter_shape[0], filter_shape[1],)
        self.weights.append(Data(dtype=self.dtype, shape=weight_shape, name='W{}'.format(self.layer_odr)))
        if self.use_bias:
            self.weights.append(Data(dtype=self.dtype, shape=(output_shape[3],), name='B{}'.format(self.layer_odr)))

        # set params
        self.set_params()

        #intialization
        c_input = open(self.template_path + "init/Conv_var_Initializer_f.txt")
        conv_input = c_input.read()
        begin = ''
        end = ''
        if self.use_bias == False:
           begin = '/*'
           end = '*/'
        func = conv_input.format(Input_channel=input_shape[3], Output_channel=output_shape[3], Filter_width=filter_shape[0], Filter_height=filter_shape[1],
            line_number=self.layer_odr, comment_begin=begin, comment_end=end)
        self.function['init'] = func + "\n\t"

        # code
        if self.use_bias:
            conv_s = open(self.template_path + "function/Conv2D_same_stride_bias.txt")
            conv_v = open(self.template_path + "function/Conv2D_valid_bias.txt")
            conv2d_same = conv_s.read()
            conv2d_valid = conv_v.read()
        else:
            conv_s = open(self.template_path + "function/Conv2D_same_stride.txt")
            conv_v = open(self.template_path + "function/Conv2D_valid.txt")
            conv2d_same = conv_s.read()
            conv2d_valid = conv_v.read()

        if self.config['padding'] == 'valid':
            func = conv2d_valid.format(Name=self.config["name"], Input_channel=input_shape[3], Input_width=input_shape[1]
                                       ,Stride_width=stride_shape[0], Stride_height=stride_shape[1],
                                       Input_height=input_shape[2], Output_channel=output_shape[3],
                                       Filter_width=filter_shape[0], Filter_height=filter_shape[1],
                                       Output_width=output_shape[1], Output_height=output_shape[2])
            self.function['code'] = func + "\n"
        else:
            func = conv2d_same.format(Name=self.config["name"], Input_channel=input_shape[3],
                                      Input_width=input_shape[1], Stride_width=stride_shape[0],
                                      Stride_height=stride_shape[1], Input_height=input_shape[2],
                                      Output_channel=output_shape[3], Filter_width=filter_shape[0],
                                      Filter_height=filter_shape[1], Output_width=output_shape[1],
                                      Output_height=output_shape[2])
            self.function['code'] = func + "\n"

class Conv2D_HW(Layers):

    def __init__(self, config={}, inputs=[], dtype='DATA_T', layer_odr=0, post=''):
        super(Conv2D_HW, self).__init__(config, inputs, dtype, layer_odr, post)

        # get shape
        input_shape = eval(self.config['batch_input_shape'])
        output_shape = eval(self.config['batch_output_shape'])
        filter_shape = eval(self.config['kernel_size'])
        stride_shape = eval(self.config['strides'])
        dilation_rate = eval(self.config['dilation_rate'])
        padding = str(self.config['padding'])
        self.use_bias = eval(self.config['use_bias'])

        # set_output
        self.set_output()

        num = self.layer_odr

        if(filter_shape[0] == 1):  # 1x1 filter
            # set weight shape
            weight_shape = (output_shape[3], input_shape[3],)
            # optimized code
            self.function['optimized_code'] += "#pragma HLS ARRAY_PARTITION variable=W{line_num}_i complete dim=1\n#pragma HLS ARRAY_PARTITION variable=B{line_num}_i complete\n".format(line_num = self.layer_odr)
            # copy values (weight)
            self.function['copy_values'] += "W"+str(num)+"_i_m_loop: for (m=0; m<"+str(output_shape[3])+"; m++) {\n  W"+str(num)+"_i_k_loop: for (k=0; k<"+str(input_shape[3])+"; k++) {\n  W"+str(num)+"_i[m][k] = W"+str(num)+"[m][k];\n  }\n}\n"
        else:
            # set_weight shape
            weight_shape=(output_shape[3], input_shape[3], filter_shape[0], filter_shape[1],)
            # optimized code
            self.function['optimized_code'] += "#pragma HLS ARRAY_PARTITION variable=W{line_num}_i complete dim=1\n#pragma HLS ARRAY_PARTITION variable=W{line_num}_i complete dim=3\n#pragma HLS ARRAY_PARTITION variable=W{line_num}_i complete dim=4\n#pragma HLS ARRAY_PARTITION variable=B{line_num}_i complete\n".format(line_num = self.layer_odr)
            # copy values (weight)
            self.function['copy_values'] += "W"+str(num)+"_i_m_loop: for (m=0; m<"+str(output_shape[3])+"; m++) {\n  W"+str(num)+"_i_k_loop: for (k=0; k<"+str(input_shape[3])+"; k++) {\n  W"+str(num)+"_i_i_loop: for (i=0; i<"+str(filter_shape[0])+"; i++) {\n  W"+str(num)+"_i_j_loop: for (j=0; j<"+str(filter_shape[1])+"; j++) {\n  W"+str(num)+"_i[m][k][i][j] = W"+str(num)+"[m][k][i][j];\n      }\n    }\n  }\n}\n"

        # set weight, Bias
        self.weights.append(Data(dtype=self.dtype, shape=weight_shape, name='W{}_i'.format(num)))
        self.weights.append(Data(dtype=self.dtype, shape=(output_shape[3],), name='B{}_i'.format(num)))

        # copy values (Bias)
        self.function['copy_values'] += "B"+str(num)+"_i_m_loop: for (m=0; m<"+str(output_shape[3])+"; m++) {\n"+"\tB"+str(num)+"_i[m] = B"+str(num)+"[m];\n}\n"

        # set params
        self.set_params()

        # code
        fname = "function/HW_conv_{}_{}x{}_{}x{}({}).txt".format(self.config['padding'], filter_shape[0],filter_shape[1],stride_shape[0],stride_shape[1],self.config['activation'])
        conv_f = open(self.template_path + fname);
        conv = conv_f.read()

        func = conv.format(Name=self.config["name"], Input_channel=input_shape[3], Input_width=input_shape[1],Stride_width=stride_shape[0], Stride_height=stride_shape[1], Input_height=input_shape[2], Output_channel=output_shape[3], Filter_width=filter_shape[0], Filter_height=filter_shape[1], Output_width=output_shape[1], Output_height=output_shape[2])
        self.function['code'] = func + "\n"

class Conv2D_DAC2017(Layers):

    def __init__(self, config={}, inputs=[], dtype='DATA_T', layer_odr=0, post=''):
        super().__init__(config, inputs, dtype, layer_odr, post)

        # get shape
        input_shape = eval(self.config['batch_input_shape'])
        output_shape = eval(self.config['batch_output_shape'])
        filter_shape = eval(self.config['kernel_size'])
        stride_shape = eval(self.config['strides'])
        dilation_rate = eval(self.config['dilation_rate'])
        padding = str(self.config['padding'])
        use_bias = eval(self.config['use_bias'])

        # set_output
        self.set_output(output_shape[1:], self.layer_odr)


        # set_weight
        weight_shape=(output_shape[3], input_shape[3], filter_shape[0], filter_shape[1])
        self.weights.append(Data(dtype=self.dtype, shape=weight_shape, name='W{}'.format(self.layer_odr)))
        if use_bias:
            self.weights.append(Data(dtype=self.dtype, shape=(output_shape[3],), name='B{}'.format(self.layer_odr)))

        # init part
        #self.weights_odr.append([3, 2, 0, 1])  # keras: kernel_width, kernel_height, input_channel, output_channel
        #if use_bias:
        #    self.weights_odr.append([0])

        # params
        #self.function['params'] = self.get_params()

        # code
        conv_s = open("src/model/template/function/Conv2D_same_DAC2017.txt")
        conv2d_same = conv_s.read()
        func = conv2d_same.format(Name=self.config["name"], Input_channel=input_shape[3],
                                  Input_width=input_shape[1], Stride_width=stride_shape[0],
                                  Stride_height=stride_shape[1], Input_height=input_shape[2],
                                  Output_channel=output_shape[3], Filter_width=filter_shape[0],
                                  Filter_height=filter_shape[1], Output_width=output_shape[1],
                                  Output_height=output_shape[2])
        self.function['code'] = func + "\n"
