from .Data import Data
from .Layers import Layers
from string import Template

class GlobalMaxPooling2D(Layers):

    def __init__(self, config={}, inputs=[], dtype='DATA_T', layer_odr=0, post=''):
        super(GlobalMaxPooling2D, self).__init__(config, inputs, dtype, layer_odr, post)

        # get shape
        input_shape = eval(self.config['batch_input_shape'])
        output_shape = eval(self.config['batch_output_shape'])

        # set_output
        self.set_output(output_shape[1:], self.layer_odr)

        # set_weight

        # init part

        # code
        mxp = open("Code_Generator/Template/Function/GlobalMaxPooling2D.txt")
        template = mxp.read()
        func = template.format(Name=self.config["name"], Input_channel=input_shape[3], Input_width=input_shape[1],
                               Input_height=input_shape[2])
        self.function['code'] = func + "\n"
