from .Data import Data
from .Layers import Layers
from string import Template


class SeparableConv2D(Layers):
    def __init__(self, config={}, inputs=[], dtype='DATA_T', layer_odr=0, post=''):

        super(SeparableConv2D, self).__init__(config, inputs, dtype, layer_odr, post)

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
        weight_shape = (input_shape[3], filter_shape[0], filter_shape[1])
        self.weights.append(Data(dtype=self.dtype, shape=weight_shape, name='W{}_1'.format(self.layer_odr)))
        weight_shape = (output_shape[3], input_shape[3])
        self.weights.append(Data(dtype=self.dtype, shape=weight_shape, name='W{}_2'.format(self.layer_odr)))

        # code

        sepconv = open("src/Model/template/Function/SeparableConv2D.txt")
        sep_conv = sepconv.read()

        func = sep_conv.format(Name=self.config["name"], Input_channel=input_shape[3], Input_width=input_shape[1]
                               , Stride_width=stride_shape[0], Stride_height=stride_shape[1], Input_height=input_shape[2],
                               Output_channel=output_shape[3], Filter_width=filter_shape[0], Filter_height=filter_shape[1],
                               Output_width=output_shape[1], Output_height=output_shape[2])
        self.function['code'] = func + "\n"
