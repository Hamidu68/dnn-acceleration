from .Data import Data
from .Layers import Layers
from string import Template

class ReLU(Layers):

    def __init__(self, config={}, inputs=[], dtype='DATA_T', layer_odr=0, post=''):
        super().__init__(config, inputs, dtype, layer_odr, post)

        # get shape
        input_shape = eval(self.config['batch_input_shape'])
        output_shape = eval(self.config['batch_output_shape'])
        max_value = int(eval(self.config['max_value']))

        # set_output
        self.set_output(output_shape[1:], self.layer_odr)

        # set_weight

        # init part

        # code
        rl = open("Code_Generator/Template/Function/Relu.txt")
        template = rl.read()
        func = template.format(Name=self.config["name"], Input_channel=input_shape[3], Input_width=input_shape[1],
                               Input_height=input_shape[2], Output_channel=output_shape[3],
                               Output_width=output_shape[1], Output_height=output_shape[2], max_value=max_value)
        self.function['code'] = func + "\n"