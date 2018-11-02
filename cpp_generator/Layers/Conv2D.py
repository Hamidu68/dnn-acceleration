from .Data import Data
from .Layers import Layers
from string import Template


class Conv2D(Layers):

    def __init__(self, config={}, inputs=[], dtype='DATA_T', layer_odr=0, post='') :

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
        if use_bias :
            self.weights.append(Data(dtype=self.dtype, shape=(output_shape[3],), name='B{}'.format(self.layer_odr)))

        # init part
        self.weights_odr.append([3, 2, 0, 1])  # keras: kernel_width, kernel_height, input_channel, output_channel
        if use_bias:
            self.weights_odr.append([0])

        # params
        self.function['params'] = self.get_params()

        # code
        conv_s = open("../Template/Function/Conv2D_same_relu.txt")
        conv_v = open("../Template/Function/Conv2D_valid.txt")
        conv2d_same = Template(conv_s.read())
        conv2d_valid = Template(conv_v.read())
        l = {'Name': self.config["name"], 'Input_channel': input_shape[3], 'Input_width': input_shape[1],
             'Stride_width': stride_shape[0], 'Stride_height': stride_shape[1],
             'Input_height': input_shape[2], 'Output_channel': output_shape[3], 'Filter_width': filter_shape[0],
             'Filter_height': filter_shape[1], 'Output_width': output_shape[1], 'Output_height': output_shape[2]}
        if self.config['padding'] == 'valid':
            self.function['code'] = conv2d_valid.substitute(l) + "\n"
        else:
            self.function['code'] = conv2d_same.substitute(l) + "\n"
