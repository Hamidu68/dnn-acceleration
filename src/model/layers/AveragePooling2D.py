from .Data import Data
from .Layers import Layers
from string import Template


class AveragePooling2D(Layers):

    def __init__(self, config={}, inputs=[], dtype='DATA_T', layer_odr=0, post=''):
        super(AveragePooling2D, self).__init__(config, inputs, dtype, layer_odr, post)

        # get shape
        input_shape = eval(self.config['batch_input_shape'])
        output_shape = eval(self.config['batch_output_shape'])
        stride_shape = eval(self.config['strides'])
        pool_shape = eval(self.config['pool_size'])

        # set_output
        self.set_output()

        # code
        avp = open(self.template_path + "function/AveragePooling2D.txt")
        template = avp.read()
        func = template.format(Name=self.config["name"], Input_channel=input_shape[3], Input_width=input_shape[1],
                               Input_height=input_shape[2], Output_channel=output_shape[3], Output_width=output_shape[1],
                               Output_height=output_shape[2], Stride_width=stride_shape[0], Stride_height=stride_shape[1],
                               Pool_width=pool_shape[0], Pool_height=pool_shape[1])
        self.function['code'] = func + "\n"
