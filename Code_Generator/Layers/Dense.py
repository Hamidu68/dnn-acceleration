from .Data import Data
from .Layers import Layers
from string import Template

class Dense(Layers):

    def __init__(self, config={}, inputs=[], dtype='DATA_T', layer_odr=0, post=''):
        super(Dense, self).__init__(config, inputs, dtype, layer_odr, post)

        # get shape
        input_shape = eval(self.config['batch_input_shape'])
        output_shape = eval(self.config['batch_output_shape'])
        use_bias = eval(self.config['use_bias'])

        # set_output
        self.set_output(output_shape[1:], self.layer_odr)

        # set_weight

        # init part

        # code
        if use_bias:
            den_s = open("Code_Generator/Template/Function/Dense_Softmax_bias.txt")
            den_r = open("Code_Generator/Template/Function/Dense_Relu_bias.txt")
            dense_softmax = den_s.read()
            dense_relu = den_r.read()
        else:
            den_s = open("Code_Generator/Template/Function/Dense_Softmax.txt")
            den_r = open("Code_Generator/Template/Function/Dense_Relu.txt")
            dense_softmax = den_s.read()
            dense_relu = den_r.read()

        if self.config['activation'] == 'relu':  # Activation = relu
            func = dense_softmax.format(Name=self.config["name"], Input_channel=input_shape[1],
                                        Output_channel=output_shape[1])
            self.function['code'] += func + "\n"
        else:  # Activation = softmax
            func = dense_relu.format(Name=self.config["name"], Input_channel=input_shape[1],
                                     Output_channel=output_shape[1])
            self.function['code'] += func + "\n"
