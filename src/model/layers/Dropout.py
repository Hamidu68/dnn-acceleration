from .Data import Data
from .Layers import Layers
from string import Template


class Dropout(Layers):

    def __init__(self, config={}, inputs=[], dtype='DATA_T', layer_odr=0, post=''):
        super(Dropout, self).__init__(config, inputs, dtype, layer_odr, post)

        # get shape
        input_shape = eval(self.config['batch_input_shape'])
        output_shape = eval(self.config['batch_output_shape'])
        rate_n = eval(self.config['rate'])

        # set_output
        self.set_output()

        # code
        if len(input_shape)==2:
            ad = open("src/Model/template/Function/Dropout1.txt")
            template = ad.read()
            func = template.format(Name=self.config['name'], Input_channel=input_shape[1], Output_channel=output_shape[1], rate=rate_n)
            self.function['code'] = func + "\n"
        else:
            ad = open("src/Model/template/Function/Dropout2.txt")
            template = ad.read()
            func = template.format(Name=self.config['name'], Input_channel=input_shape[3], Input_width=input_shape[1],
                                   Input_height=input_shape[2], Output_channel=output_shape[3],
                                   Output_width=output_shape[1], Output_height=output_shape[2], rate=rate_n)
            self.function['code'] = func + "\n"
