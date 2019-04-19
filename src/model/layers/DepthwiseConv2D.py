from .Data import Data
from .Layers import Layers
from string import Template


class DepthwiseConv2D(Layers):

    def __init__(self, config={}, inputs=[], dtype='DATA_T', layer_odr=0, post=''):
        super(DepthwiseConv2D, self).__init__(config, inputs, dtype, layer_odr, post)

        # get shape
        input_shape = eval(self.config['batch_input_shape'])
        output_shape = eval(self.config['batch_output_shape'])
        filter_shape = eval(self.config['kernel_size'])
        stride_shape = eval(self.config['strides'])
        padding = str(self.config['padding'])
        self.use_bias = eval(self.config['use_bias'])

        # set_output
        self.set_output()

        # set_weight
        weight_shape = (output_shape[3], filter_shape[0], filter_shape[1],)
        self.weights.append(Data(dtype=self.dtype, shape=weight_shape, name='W{}'.format(self.layer_odr)))
        if self.use_bias:
            self.weights.append(Data(dtype=self.dtype, shape=(output_shape[3],), name='B{}'.format(self.layer_odr)))

        # set_params
        self.set_params()

        # initialization
        dc_input = open(self.template_path + "init/depthConv_var_Initializer_f.txt")
        dconv_input = dc_input.read()
        begin = ''
        end = ''
        if self.use_bias == False:
            begin = '/*'
            end = '*/'
        func = dconv_input.format(Output_channel=output_shape[3], Filter_width=filter_shape[0],
                                          Filter_height=filter_shape[1], line_number=l_n,
                                          comment_begin=begin, comment_end=end)
        self.function['init'] = func + "\n\t"

        # code
        dconv_s = open(self.template_path + "function/DepthwiseConv2D_same.txt")
        dconv_v = open(self.template_path + "function/DepthwiseConv2D_valid.txt")
        dconv2d_s = dconv_s.read()
        dconv2d_v = dconv_v.read()
        if padding == 'valid':
            func = dconv2d_v.format(Name=self.config["name"], Input_channel=input_shape[3],
                                    Input_width=input_shape[1]
                                    , Stride_width=stride_shape[0], Stride_height=stride_shape[1],
                                    Input_height=input_shape[2], Output_channel=output_shape[3],
                                    Filter_width=filter_shape[0], Filter_height=filter_shape[1],
                                    Output_width=output_shape[1], Output_height=output_shape[2])
            self.function['code'] = func + "\n"
        else:
            func = dconv2d_s.format(Name=self.config["name"], Input_channel=input_shape[3],
                                    Input_width=input_shape[1], Stride_width=stride_shape[0],
                                    Stride_height=stride_shape[1], Input_height=input_shape[2],
                                    Output_channel=output_shape[3], Filter_width=filter_shape[0],
                                    Filter_height=filter_shape[1], Output_width=output_shape[1],
                                    Output_height=output_shape[2])
            self.function['code'] = func + "\n"
