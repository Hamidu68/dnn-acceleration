from .Data import Data
from .Layers import Layers
from string import Template

class InputLayer(Layers):
    def __init__(self, config={}, inputs=[], dtype='DATA_T', layer_odr=0, post=''):
        super(InputLayer,self).__init__(config, inputs, dtype, layer_odr, post)

        input_shape = eval(config['batch_input_shape'])
        output_shape = eval(config['batch_output_shape'])

        #set input
        shape = (input_shape[1],input_shape[2],input_shape[3],)
        self.inputs.append(Data(dtype=self.dtype, shape=shape, name='I'))

        #set output
        #self.set_output()

        # set params
        self.function['static_w'] = inputs[0].get_static_variable()
        self.function['call_p'] += inputs[0].get_call_param()
        self.function['func_p'] += inputs[0].get_func_param()

        #initialization
        i_input = open(self.template_path + "init/Input_var_Initializer.txt")
        init_input = i_input.read()
        func = init_input.format(Input_channel=input_shape[3], Input_width=input_shape[1], Input_height=input_shape[2])
        self.function['init'] = func + "\n\t"

class InputLayer_HW(Layers):
    def __init__(self, config={}, inputs=[], dtype='DATA_T', layer_odr=0, post=''):
        super(InputLayer_HW,self).__init__(config, inputs, dtype, layer_odr, post)

        input_shape = eval(config['batch_input_shape'])

        #set input
        shape = (input_shape[1],input_shape[2],input_shape[3],)
        self.inputs.append(Data(dtype=self.dtype, shape=shape, name='I_i'))

        # set params
        self.function['static_w'] = inputs[0].get_static_variable()
        self.function['call_p'] += inputs[0].get_call_param()
        self.function['func_p'] += inputs[0].get_func_param()

        # copy values
        self.function['copy_values'] += "\tI_i_k_loop: for (k=0; k<"+str(input_shape[3])+"; k++) {\n  I_i_x_loop: for (x=0; x<"+str(input_shape[1])+"; x++) {\n  I_i_y_loop: for (y=0; y<"+str(input_shape[2])+"; y++) {\n  I_i[k][x][y] = I[k][x][y];\n    }\n  }\n}\n"