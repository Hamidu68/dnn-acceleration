from .Data import Data
from .Layers import Layers
from string import Template

class Concatenate(Layers):

    def __init__(self, config={}, inputs=[], dtype='DATA_T', layer_odr=0, post=''):
        super(Concatenate, self).__init__(config, inputs, dtype, layer_odr, post)

        # get shape
        output_shape = eval(self.config['batch_output_shape'])
        input_shape = eval(self.config['batch_input_shape'])

        # set_output
        self.set_output()

        # code
        con2 = open(self.template_path + "function/Concatenate2.txt")
        con3 = open(self.template_path + "function/Concatenate3.txt")
        con4 = open(self.template_path + "function/Concatenate4.txt")

        con2_r = con2.read()
        con3_r = con3.read()
        con4_r = con4.read()

        if len(input_shape) == 2:
            func = con2_r.format(Name=self.config['name'], Input_channel1=input_shape[0][3],
                                 Input_channel2=input_shape[1][3], Output_channel=output_shape[3],
                                 Output_width=output_shape[1], Output_height=output_shape[2])
        elif len(input_shape) == 3:
            func = con3_r.format(Name=self.config['name'], Input_channel1=input_shape[0][3],
                                 Input_channel2=input_shape[1][3], Input_channel3=input_shape[2][3],
                                 Output_channel=output_shape[3], Output_width=output_shape[1],
                                 Output_height=output_shape[2])
        elif len(input_shape) == 4:
            func = con4_r.format(Name=self.config['name'], Input_channel1=input_shape[0][3],
                                 Input_channel2=input_shape[1][3], Input_channel3=input_shape[2][3],
                                 Input_channel4=input_shape[3][3], Output_channel=output_shape[3],
                                 Output_width=output_shape[1], Output_height=output_shape[2])

        self.function['code'] = func + "\n"
