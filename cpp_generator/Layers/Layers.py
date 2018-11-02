from .Data import Data
from string import Template


class Layers:
    def __init__(self, config={}, inputs=[], dtype='DATA_T', layer_odr=0, post=''):
        self.config = config
        self.inputs = inputs
        self.dtype = dtype
        self.layer_odr = layer_odr
        self.post = post
        self.output = None
        self.weights = []
        self.weights_odr = []
        self.function = {'name': self.config['name']+post,
                         'params': [],
                         'code': ''}  

    def set_output(self, shape=(), odr=0):
        self.output = Data(dtype=self.dtype, shape=shape, name='O{}'.format(odr))
    '''
    def set_layer(self):
        return
    '''


class Conv2D_HW(Layers):

    def __init__(self, config={}, inputs=[], dtype='DATA_T', layer_odr=0, post=''):
        super().__init__(config, inputs, dtype, layer_odr, post)

        # get shape
        input_shape = eval(self.config['batch_input_shape'])
        output_shape = eval(self.config['batch_output_shape'])
        kernel_size = eval(self.config['kernel_size'])
        strides = eval(self.config['strides'])
        dilation_rate = eval(self.config['dilation_rate'])
        padding = str(self.config['padding'])
        use_bias = eval(self.config['use_bias'])
        
        # set_output
        self.set_output(output_shape[1:], self.layer_odr)

        # set_weight
        weight_shape=(output_shape[3],input_shape[3],kernel_size[0],kernel_size[1])
        self.weights.append(Data(dtype=self.dtype, shape=weight_shape, name='W{}'.format(self.layer_odr)))
        if use_bias :
            self.weights.append(Data(dtype=self.dtype, shape=(output_shape[3],), name='B{}'.format(self.layer_odr)))

        # init part
        self.weights_odr.append([3,2,0,1])  # keras: kernel_width, kernel_height, input_channel, output_channel
        if use_bias :
            self.weights_odr.append([0])

        # code
        self.function['code'] = self.template.format(name='',
                                                     post=self.post,
                                                     params='',
                                                     inner='')


class Conv2D_DAC2017(Layers):

    def __init__(self, config={}, inputs=[], dtype='DATA_T', layer_odr=0, post=''):
        super().__init__(config, inputs, dtype, layer_odr, post)

        # get shape
        input_shape = eval(self.config['batch_input_shape'])
        output_shape = eval(self.config['batch_output_shape'])
        filter_shape = eval(self.config['kernel_size'])
        stride_shape = eval(self.config['strides'])
        dilation_rate = eval(self.config['dilation_rate'])
        padding = str(self.config['padding'])
        use_bias = eval(self.config['use_bias'])
        
        # set_output
        self.set_output(output_shape[1:], self.layer_odr)

        # set_weight
        weight_shape=(output_shape[3],input_shape[3],filter_shape[0],filter_shape[1])
        self.weights.append(Data(dtype=self.dtype, shape=weight_shape, name='W{}'.format(self.layer_odr)))
        if use_bias :
            self.weights.append(Data(dtype=self.dtype, shape=(output_shape[3],), name='B{}'.format(self.layer_odr)))

        # init part
        self.weights_odr.append([3, 2, 0, 1])  # keras: kernel_width, kernel_height, input_channel, output_channel
        if use_bias :
            self.weights_odr.append([0])

        # code
        # self.function['code']=

        
class MaxPooling2D(Layers):

    def __init__(self, config={}, inputs=[], dtype='DATA_T', layer_odr=0, post=''):
        super().__init__(config, inputs, dtype, layer_odr, post)

        # get shape
        input_shape = eval(self.config['batch_input_shape'])
        output_shape = eval(self.config['batch_output_shape'])
        pool_shape = eval(self.config['pool_size'])
        stride_shape = eval(self.config['strides'])

        # set_output
        self.set_output(output_shape[1:], self.layer_odr)

        # set_weight

        # init part

        # code
        mxp = open("cpp_generator/Template/Function/MaxPooling2D.txt")
        mxpool2d = Template(mxp.read())
        l = {'Name': self.config["name"], 'Input_channel': input_shape[3], 'Input_width': input_shape[1],
             'Input_height': input_shape[2], 'Output_channel': output_shape[3],
             'Output_width': output_shape[1], 'Output_height': output_shape[2], 'Stride_width': stride_shape[0],
             'Stride_height': stride_shape[1],
             'Pool_width': pool_shape[0], 'Pool_height': pool_shape[1]}
        self.function['code']=mxpool2d.substitute(l) + "\n"


class MaxPooling2D_HW(Layers):

    def __init__(self, config={}, inputs=[], dtype='DATA_T', layer_odr=0, post=''):
        super().__init__(config, inputs, dtype, layer_odr, post)

        # get shape
        input_shape = eval(self.config['batch_input_shape'])
        output_shape = eval(self.config['batch_output_shape'])

        # set_output
        self.set_output(output_shape[1:], self.layer_odr)

        # set_weight

        # init part

        # code
        # self.function['code']

        
class BatchNormalization(Layers):

    def __init__(self, config={}, inputs=[], dtype='DATA_T', layer_odr=0, post=''):
        super().__init__(config, inputs, dtype, layer_odr, post)

        # get shape
        input_shape = eval(self.config['batch_input_shape'])
        output_shape = eval(self.config['batch_output_shape'])

        # set_output
        self.set_output(output_shape[1:], self.layer_odr)

        # set_weight

        # init part

        # code
        batch_normal = open("cpp_generator/Template/Function/BatchNormalization.txt")
        batchnorm = Template(batch_normal.read())
        l = {'Name': self.config['name'], 'Input_channel': input_shape[3], 'Input_width': input_shape[1],
             'Input_height': input_shape[2], 'Output_channel': output_shape[3],
             'Output_width': output_shape[1], 'Output_height': output_shape[2]}
        self.function['code'] = batchnorm.substitute(l) + "\n"


class Activation(Layers):

    def __init__(self, config={}, inputs=[], dtype='DATA_T', layer_odr=0, post=''):
        super().__init__(config, inputs, dtype, layer_odr, post)

        # get shape
        input_shape = eval(self.config['batch_input_shape'])
        output_shape = eval(self.config['batch_output_shape'])

        # set_output
        self.set_output(output_shape[1:], self.layer_odr)

        # set_weight

        # init part

        # code
        rl = open("cpp_generator/Template/Function/Relu.txt")
        relu = Template(rl.read())
        l = {'Name': self.config["name"], 'Input_channel': input_shape[3], 'Input_width': input_shape[1],
             'Input_height': input_shape[2], 'Output_channel': output_shape[3],
             'Output_width': output_shape[1], 'Output_height': output_shape[2]}
        self.function['code']=relu.substitute(l) + "\n"


class AveragePooling2D(Layers):

    def __init__(self, config={}, inputs=[], dtype='DATA_T', layer_odr=0, post=''):
        super().__init__(config, inputs, dtype, layer_odr, post)

        # get shape
        input_shape = eval(self.config['batch_input_shape'])
        output_shape = eval(self.config['batch_output_shape'])
        stride_shape = eval(self.config['strides'])
        pool_shape = eval(self.config['pool_size'])

        # set_output
        self.set_output(output_shape[1:], self.layer_odr)

        # set_weight

        # init part

        # code
        avp = open("cpp_generator/Template/Function/AveragePooling2D.txt")
        avepooling2d = Template(avp.read())
        l = {'Name': self.config["name"], 'Input_channel': input_shape[3], 'Input_width': input_shape[1],
             'Input_height': input_shape[2], 'Output_channel': output_shape[3],
             'Output_width': output_shape[1], 'Output_height': output_shape[2], 'Stride_width': stride_shape[0],
             'Stride_height': stride_shape[1],
             'Pool_width': pool_shape[0], 'Pool_height': pool_shape[1]}
        self.function['code']=avepooling2d.substitute(l) + "\n"


class ZeroPadding2D(Layers):

    def __init__(self, config={}, inputs=[], dtype='DATA_T', layer_odr=0, post=''):
        super().__init__(config, inputs, dtype, layer_odr, post)

        # get shape
        input_shape = eval(self.config['batch_input_shape'])
        output_shape = eval(self.config['batch_output_shape'])
        padding = eval(self.config['padding'])

        # set_output
        self.set_output(output_shape[1:], self.layer_odr)

        # set_weight

        # init part

        # code
        zp = open("cpp_generator/Template/Function/ZeroPadding.txt")
        zeropad2d = Template(zp.read())
        l = {'Name': self.config["name"], 'Input_channel': input_shape[3], 'Input_width': input_shape[1],
             'Input_height': input_shape[2], 'Output_channel': output_shape[3],
             'Output_width': output_shape[1], 'Output_height': output_shape[2], 'Padding_size': padding[0][0]}
        self.function['code']=zeropad2d.substitute(l) + "\n"


class Flatten(Layers):

    def __init__(self, config={}, inputs=[], dtype='DATA_T', layer_odr=0, post=''):
        super().__init__(config, inputs, dtype, layer_odr, post)

        # get shape
        input_shape = eval(self.config['batch_input_shape'])
        output_shape = eval(self.config['batch_output_shape'])

        # set_output
        self.set_output(output_shape[1:], self.layer_odr)

        # set_weight

        # init part

        # code
        fla = open("cpp_generator/Template/Function/Flatten.txt")
        flatten = Template(fla.read())
        l = {'Name': self.config["name"], 'Input_channel': input_shape[3], 'Input_width': input_shape[1],
             'Input_height': input_shape[2], 'Output_channel': output_shape[1]}
        self.function['code']=flatten.substitute(l) + "\n"


class Dense(Layers):

    def __init__(self, config={}, inputs=[], dtype='DATA_T', layer_odr=0, post=''):
        super().__init__(config, inputs, dtype, layer_odr, post)

        # get shape
        input_shape = eval(self.config['batch_input_shape'])
        output_shape = eval(self.config['batch_output_shape'])

        # set_output
        self.set_output(output_shape[1:], self.layer_odr)

        # set_weight

        # init part

        # code
        den_s = open("cpp_generator/Template/Function/Dense_Softmax.txt")
        den_r = open("cpp_generator/Template/Function/Dense_Relu.txt")
        dense_softmax = Template(den_s.read())
        dense_relu = Template(den_r.read())
        l = {'Name': self.config["name"], 'Input_channel': input_shape[1], 'Output_channel': output_shape[1]}
        if self.config['activation'] == 'relu':  # Activation = relu
            self.function['code'] += dense_relu.substitute(l) + "\n"
        else:  # Activation = softmax
            self.function['code'] += dense_softmax.substitute(l) + "\n"

        
class Add(Layers):

    def __init__(self, config={}, inputs=[], dtype='DATA_T', layer_odr=0, post=''):
        super().__init__(config, inputs, dtype, layer_odr, post)

        # get shape
        output_shape = eval(self.config['batch_output_shape'])

        # set_output
        self.set_output(output_shape[1:], self.layer_odr)

        # set_weight

        # init part

        # code
        ad = open("cpp_generator/Template/Function/Add.txt")
        add = Template(ad.read())
        l = {'Name': self.config['name'], 'Input_channel1': output_shape[3], 'Input_width1': output_shape[1],
             'Input_height1': output_shape[2], 'Input_channel2': output_shape[3], 'Input_width2': output_shape[1],
             'Input_height2': output_shape[2], 'Output_channel': output_shape[3], 'Output_width': output_shape[1],
             'Output_height': output_shape[2]}
        self.function['code']=add.substitute(l) + "\n"