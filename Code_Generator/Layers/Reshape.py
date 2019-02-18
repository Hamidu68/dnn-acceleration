from .Data import Data
from .Layers import Layers
from string import Template

class Reshape(Layers):

    def __init__(self, config={}, inputs=[], dtype='DATA_T', layer_odr=0, post=''):
        super(Reshape, self).__init__(config, inputs, dtype, layer_odr, post)

        # get shape
        input_shape = eval(self.config['batch_input_shape'])
        output_shape = eval(self.config['batch_output_shape'])

        # set_output
        self.set_output(output_shape[1:], self.layer_odr)

        # set_weight

        # init part

        # code
        if len(input_shape)>=3:
            rl = open("Code_Generator/Template/Function/Reshape1.txt")
            template = rl.read()
            func = template.format(Name=self.config["name"], Input_channel=input_shape[3],
                                   Output_channel=output_shape[1])
        else:
            rl = open("Code_Generator/Template/Function/Reshape2.txt")
            template = rl.read()
            func = template.format(Name=self.config["name"], Input_channel=input_shape[1],
                                   Output_channel=output_shape[3])

        self.function['code'] = func + "\n"
