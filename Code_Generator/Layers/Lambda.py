from .Data import Data
from .Layers import Layers
from string import Template

class Lambda(Layers):

    def __init__(self, config={}, inputs=[], dtype='DATA_T', layer_odr=0, post=''):
        super().__init__(config, inputs, dtype, layer_odr, post)

        # get shape
        output_shape = eval(self.config['batch_output_shape'])
        scale_dict = ast.literal_eval(self.config['arguments'])
        scale = scale_dict['scale']
        # set_output
        self.set_output(output_shape[1:], self.layer_odr)

        # set_weight

        # init part

        # code
        lamda = open("Code_Generator/Template/Function/Lambda.txt")
        lamda_r = lamda.read()

        func = lamda_r.format(Name=self.config['name'], Output_channel=output_shape[3], scale=scale,
                              Output_width=output_shape[1], Output_height=output_shape[2])
        self.function['code'] = func + "\n"