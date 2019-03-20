from .Data import Data
from .Layers import Layers
from string import Template

class Cropping2D(Layers):
    def __init__(self, config={}, inputs=[], dtype='DATA_T', layer_odr=0, post=''):

        super(Cropping2D, self).__init__(config, inputs, dtype, layer_odr, post)

        # get shape
        input_shape = eval(self.config['batch_input_shape'])
        output_shape = eval(self.config['batch_output_shape'])
        cropping = eval(self.config['cropping'])

        # set_output
        self.set_output(output_shape[1:], self.layer_odr)

        # code
        cropp = open("Code_Generator/Template/Function/Cropping2D.txt")
        cropp_2d = cropp.read()

        func = cropp_2d.format(Name=self.config["name"], Input_channel=input_shape[3], Input_width=input_shape[1],
                               Input_height=input_shape[2], Output_channel=output_shape[3],
                               Output_width=output_shape[1], Output_height=output_shape[2],
                               top=cropping[0][0], bottom=cropping[0][1], left=cropping[1][0], right=cropping[1][1])
        self.function['code'] = func + "\n"
