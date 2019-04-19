from .Data import Data
from .Layers import Layers
from string import Template

class BatchNormalization(Layers):

    def __init__(self, config={}, inputs=[], dtype='DATA_T', layer_odr=0, post=''):
        super(BatchNormalization, self).__init__(config, inputs, dtype, layer_odr, post)

        # get shape
        input_shape = eval(self.config['batch_input_shape'])
        output_shape = eval(self.config['batch_output_shape'])

        if self.config['scale'] == 'False':
            num = 3
        else :
            num = 4

        # set_output
        self.set_output()

        # set_weight
        weight_shape = (num, output_shape[3],)
        self.weights.append(Data(dtype=self.dtype, shape=weight_shape, name='W{}'.format(self.layer_odr)))

        # set params
        self.set_params()

        # initialization
        b_input = open(self.template_path + "init/Batch_var_Initializer_f.txt")
        batch_input = b_input.read()
        func = batch_input.format(Output_channel=(output_shape[3]), line_number=l_n, num=num)
        self.function['init'] = func + "\n\t"

        # code
        if self.config['scale'] == 'False' :
            batch_normal = open(self.template_path + "function/BatchNormalization_no_scale.txt")
        else :
            batch_normal = open(self.template_path + "function/BatchNormalization.txt")
        template = batch_normal.read()
        func = template.format(Name=self.config['name'], Input_channel= input_shape[3], Input_width= input_shape[1],
                               Input_height=input_shape[2], Output_channel=output_shape[3], Output_width=output_shape[1]
                               , Output_height=output_shape[2], epsilon=self.config['epsilon'],
                               momentum=self.config['momentum'])
        self.function['code'] = func + "\n"
