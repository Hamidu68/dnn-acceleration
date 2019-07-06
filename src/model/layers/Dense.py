from .Data import Data
from .Layers import Layers
from string import Template

class Dense(Layers):

    def __init__(self, config={}, inputs=[], dtype='DATA_T', layer_odr=0, post=''):
        super(Dense, self).__init__(config, inputs, dtype, layer_odr, post)

        # get shape
        input_shape = eval(self.config['batch_input_shape'])
        output_shape = eval(self.config['batch_output_shape'])
        self.use_bias = eval(self.config['use_bias'])

        # set_output
        self.set_output()

        # set_weight
        weight_shape=(output_shape[1], input_shape[1],)
        self.weights.append(Data(dtype=self.dtype, shape=weight_shape, name='W{}'.format(self.layer_odr)))
        if self.use_bias:
            self.weights.append(Data(dtype=self.dtype, shape=(output_shape[1],), name='B{}'.format(self.layer_odr)))

        # set_params
        self.set_params()

        #intialization
        d_input = open(self.template_path + "init/Dense_var_Initializer.txt")
        dense_input = d_input.read()
        begin = ''
        end = ''
        if self.use_bias == False :
            begin = '/*'
            end = '*/'
        func = dense_input.format(Input_channel=input_shape[1], Output_channel=output_shape[1],
                                          line_number=self.layer_odr, comment_begin=begin, comment_end=end)
        self.function['init'] = func + "\n\t"

        # code
        if self.use_bias:
            den_s = open(self.template_path + "function/Dense_Softmax_bias.txt")
            den_r = open(self.template_path + "function/Dense_Relu_bias.txt")
            dense_softmax = den_s.read()
            dense_relu = den_r.read()
        else:
            den_s = open(self.template_path + "function/Dense_Softmax.txt")
            den_r = open(self.template_path + "function/Dense_Relu.txt")
            dense_softmax = den_s.read()
            dense_relu = den_r.read()

        if self.config['activation'] == 'relu':  # Activation = relu
            func = dense_relu.format(Name=self.config["name"], Input_channel=input_shape[1],
                                        Output_channel=output_shape[1])
            self.function['code'] += func + "\n"
        else:  # Activation = softmax
            func = dense_softmax.format(Name=self.config["name"], Input_channel=input_shape[1],
                                     Output_channel=output_shape[1])
            self.function['code'] += func + "\n"

class Dense_HW(Layers):

    def __init__(self, config={}, inputs=[], dtype='DATA_T', layer_odr=0, post=''):
        super(Dense_HW, self).__init__(config, inputs, dtype, layer_odr, post)

        # get shape
        input_shape = eval(self.config['batch_input_shape'])
        output_shape = eval(self.config['batch_output_shape'])
        self.use_bias = eval(self.config['use_bias'])
        # set_output
        self.set_output()

        weight_shape=(output_shape[1], input_shape[1],)
        # optimized code
        self.function['optimized_code'] += "#pragma HLS ARRAY_PARTITION variable=W{line_num}_i complete dim=1\n#pragma HLS ARRAY_PARTITION variable=B{line_num}_i complete\n".format(line_num = self.layer_odr)

        num = self.layer_odr;
        # copy values (weight, bias)
        self.function['copy_values'] += "W"+str(num)+"_i_m_loop: for (m=0; m<"+str(output_shape[1])+"; m++) {\n  W"+str(num)+"_i_k_loop: for (k=0; k<"+str(input_shape[1])+"; k++) {\n  W"+str(num)+"_i[m][k] = W"+str(num)+"[m][k];\n      }\n}\n"
        self.function['copy_values'] += "B"+str(num)+"_i_m_loop: for (m=0; m<"+str(output_shape[1])+"; m++) {\n"+"\tB"+str(num)+"_i[m] = B"+str(num)+"[m];\n}\n"
        # set_weight, bias
        weight_shape=(output_shape[1], input_shape[1],)
        self.weights.append(Data(dtype=self.dtype, shape=weight_shape, name='W{}_i'.format(self.layer_odr)))
        self.weights.append(Data(dtype=self.dtype, shape=(output_shape[1],), name='B{}_i'.format(self.layer_odr)))

        # set_params
        self.set_params()

        # code
        den_s = open(self.template_path + "function/HW_dense_softmax_bias.txt")
        den_r = open(self.template_path + "function/HW_dense_relu_bias.txt")
        dense_softmax = den_s.read()
        dense_relu = den_r.read()

        if self.config['activation'] == 'relu':  # Activation = relu
            func = dense_relu.format(Name=self.config["name"], Input_channel=input_shape[1],
                                        Output_channel=output_shape[1])
            self.function['code'] += func + "\n"
        else:  # Activation = softmax
            func = dense_softmax.format(Name=self.config["name"], Input_channel=input_shape[1],
                                     Output_channel=output_shape[1])
            self.function['code'] += func + "\n"
