from .Data import Data
from .Layers import Layers
from string import Template


class Activation(Layers):

    def __init__(self, config={}, inputs=[], dtype='DATA_T', layer_odr=0, post=''):
        super(Activation, self).__init__(config, inputs, dtype, layer_odr, post)

        # get shape
        input_shape = eval(self.config['batch_input_shape'])
        output_shape = eval(self.config['batch_output_shape'])

        # set_output
        self.set_output()

        # code
        if self.config['activation'] == 'relu':
            rl = open(self.template_path + "function/Activation_relu.txt")
            template = rl.read()
            func = template.format(Name=self.config["name"], Input_channel= input_shape[3],
                                   Input_width= input_shape[1], Input_height=input_shape[2],
                                   Output_channel=output_shape[3], Output_width=output_shape[1],
                                   Output_height=output_shape[2])
            self.function['code'] = func + "\n"
        elif self.config['activation'] == 'softmax':
            rl = open(self.template_path + "function/Activation_softmax.txt")
            template = rl.read()
            func = template.format(Name=self.config["name"], Output_channel=output_shape[3],
                                   Input_channel=input_shape[3])
            self.function['code'] = func + "\n"

class Activation_HW(Layers):

    def __init__(self, config={}, inputs=[], dtype='DATA_T', layer_odr=0, post=''):
        super(Activation_HW, self).__init__(config, inputs, dtype, layer_odr, post)

        # get shape
        input_shape = eval(self.config['batch_input_shape'])
        output_shape = eval(self.config['batch_output_shape'])

        # set_output
        self.set_output()

        # code
        if self.config['activation'] == 'relu':
            rl = open(self.template_path + "function/HW_relu.txt")
            template = rl.read()
            func = template.format(Name=self.config["name"], Input_channel= input_shape[3],
                                   Input_width= input_shape[1], Input_height=input_shape[2])
            self.function['code'] = func + "\n"
        elif self.config['activation'] == 'softmax':
            rl = open(self.template_path + "function/HW_softmax.txt")
            template = rl.read()
            func = template.format(Name=self.config["name"], Output_channel=output_shape[3],
                                   Input_channel=input_shape[3])
            self.function['code'] = func + "\n"
