from .Data import Data
from string import Template


class Layers(object):
    def __init__(self, config={}, inputs=[], dtype='DATA_T', layer_odr=0, post=''):
        self.config = config
        self.inputs = inputs
        self.dtype = dtype
        self.layer_odr = layer_odr
        self.post = post
        self.output = None
        self.weights = []
        self.use_bias = False
        self.template_path = "src/model/template/"
        self.function = {'name': post+'_'+self.config['name'], 'print_result': '', 'copy_values': '', 'code': '','static_w': '','static_o':'','call_p': '','func_p': '', 'init': '', 'optimized_code':'', 'stream_def':''}

    def set_output(self):
        output_shape = eval(self.config['batch_output_shape'])
        if len(output_shape) <= 2:
            self.output = Data(dtype=self.dtype, shape=(output_shape[1],), name='O{}_{}'.format(self.layer_odr, self.post))
        else:
            self.output = Data(dtype=self.dtype, shape=(output_shape[1],output_shape[2],output_shape[3],), name='O{}_{}'.format(self.layer_odr, self.post))
        self.function['static_o'] = self.output.get_static_variable()

    def print_result(self):
        output_shape = eval(self.config['batch_output_shape'])
        layer = self.config['layer_type']
        o1 = open(self.template_path + "print/Print_Output3D.txt")
        o2 = open(self.template_path + "print/Print_Output1D.txt")
        output3d = o1.read()
        output1d = o2.read()
        if self.layer_odr == 0:
            output_name = 'I'
        else:
            output_name = 'O{}_{}'.format(self.layer_odr,self.post)

        if len(output_shape) <= 2:
            func = output1d.format(Name=layer+self.config['name'], first=output_shape[1], output=output_name)
        else:
            func = output3d.format(Name=layer+self.config['name'], third=output_shape[3], second=output_shape[2],first=output_shape[1], output=output_name)
        self.function['print_result'] = func+"\n\t"

    def set_params(self):
        for wei in self.weights:
            self.function['static_w'] += wei.get_static_variable()
            self.function['call_p'] += wei.get_call_param()
            self.function['func_p'] += wei.get_func_param()

    def set_output_name(self,name=''):
        self.output.set_name(name)
        self.function['static_o'] = self.output.get_static_variable()