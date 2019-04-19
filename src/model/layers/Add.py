from .Data import Data
from .Layers import Layers
from string import Template

class Add(Layers):

    def __init__(self, config={}, inputs=[], dtype='DATA_T', layer_odr=0, post=''):
        super(Add, self).__init__(config, inputs, dtype, layer_odr, post)

        # get shape
        output_shape = eval(self.config['batch_output_shape'])

        # set_output
        self.set_output()

        # code
        ad = open(self.template_path + "function/Add.txt")
        template = ad.read()
        func = template.format(Name=self.config['name'], Input_channel1=output_shape[3], Input_width1=output_shape[1],
                               Input_height1=output_shape[2], Input_channel2=output_shape[3],
                               Input_width2=output_shape[1], Input_height2=output_shape[2],
                               Output_channel=output_shape[3], Output_width=output_shape[1],
                               Output_height=output_shape[2])
        self.function['code'] = func + "\n"
