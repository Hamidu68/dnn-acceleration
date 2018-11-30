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
        self.function = {'name': post+'_'+self.config['name'],
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
        weight_shape=(output_shape[3], input_shape[3], kernel_size[0], kernel_size[1])
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
        if self.config['scale'] == 'False' :
            batch_normal = open("cpp_generator/mobilenetv2/Template/Function/BatchNormalization_no_scale.txt")
        else :
            batch_normal = open("cpp_generator/mobilenetv2/Template/Function/BatchNormalization.txt")

        template = batch_normal.read()
        func = template.format(Name=self.config['name'], Input_channel= input_shape[3], Input_width= input_shape[1],
                               Input_height=input_shape[2], Output_channel=output_shape[3], Output_width=output_shape[1]
                               , Output_height=output_shape[2], epsilon=self.config['epsilon'],
                               momentum=self.config['momentum'])
        self.function['code'] = func + "\n"


class DepthwiseConv2D(Layers):

    def __init__(self, config={}, inputs=[], dtype='DATA_T', layer_odr=0, post=''):
        super().__init__(config, inputs, dtype, layer_odr, post)

        # get shape
        input_shape = eval(self.config['batch_input_shape'])
        output_shape = eval(self.config['batch_output_shape'])
        filter_shape = eval(self.config['kernel_size'])
        stride_shape = eval(self.config['strides'])
        padding = str(self.config['padding'])
        use_bias = eval(self.config['use_bias'])

        # set_output
        self.set_output(output_shape[1:], self.layer_odr)

        # set_weight
        weight_shape = (output_shape[3], filter_shape[0], filter_shape[1])
        self.weights.append(Data(dtype=self.dtype, shape=weight_shape, name='W{}'.format(self.layer_odr)))
        if use_bias:
            self.weights.append(Data(dtype=self.dtype, shape=(output_shape[3],), name='B{}'.format(self.layer_odr)))

        # code
        dconv_s = open("cpp_generator/mobilenetv2/Template/Function/DepthwiseConv2D_same.txt")
        dconv_v = open("cpp_generator/mobilenetv2/Template/Function/DepthwiseConv2D_valid.txt")

        dconv2d_s = dconv_s.read()
        dconv2d_v = dconv_v.read()

        if padding == 'valid':
            func = dconv2d_v.format(Name=self.config["name"], Input_channel=input_shape[3],
                                    Input_width=input_shape[1]
                                    , Stride_width=stride_shape[0], Stride_height=stride_shape[1],
                                    Input_height=input_shape[2], Output_channel=output_shape[3],
                                    Filter_width=filter_shape[0], Filter_height=filter_shape[1],
                                    Output_width=output_shape[1], Output_height=output_shape[2])
            self.function['code'] = func + "\n"
        else:
            func = dconv2d_s.format(Name=self.config["name"], Input_channel=input_shape[3],
                                    Input_width=input_shape[1], Stride_width=stride_shape[0],
                                    Stride_height=stride_shape[1], Input_height=input_shape[2],
                                    Output_channel=output_shape[3], Filter_width=filter_shape[0],
                                    Filter_height=filter_shape[1], Output_width=output_shape[1],
                                    Output_height=output_shape[2])
            self.function['code'] = func + "\n"


class ReLU(Layers):

    def __init__(self, config={}, inputs=[], dtype='DATA_T', layer_odr=0, post=''):
        super().__init__(config, inputs, dtype, layer_odr, post)

        # get shape
        input_shape = eval(self.config['batch_input_shape'])
        output_shape = eval(self.config['batch_output_shape'])
        max_value = int(self.config['max_value'])

        # set_output
        self.set_output(output_shape[1:], self.layer_odr)

        # set_weight

        # init part

        # code
        rl = open("cpp_generator/mobilenetv2/Template/Function/Relu.txt")
        template = rl.read()
        func = template.format(Name=self.config["name"], Input_channel=input_shape[3], Input_width=input_shape[1],
                               Input_height=input_shape[2], Output_channel=output_shape[3],
                               Output_width=output_shape[1], Output_height=output_shape[2], max_value=max_value)
        self.function['code'] = func + "\n"


class GlobalAveragePooling(Layers):

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
        mxp = open("cpp_generator/mobilenetv2/Template/Function/GlobalAveragePooling.txt")
        template = mxp.read()
        func = template.format(Name=self.config["name"], Input_channel=input_shape[3], Input_width=input_shape[1],
                               Input_height=input_shape[2])
        self.function['code'] = func + "\n"


class Dense(Layers):

    def __init__(self, config={}, inputs=[], dtype='DATA_T', layer_odr=0, post=''):
        super().__init__(config, inputs, dtype, layer_odr, post)

        # get shape
        input_shape = eval(self.config['batch_input_shape'])
        output_shape = eval(self.config['batch_output_shape'])
        use_bias = eval(self.config['use_bias'])

        # set_output
        self.set_output(output_shape[1:], self.layer_odr)

        # set_weight

        # init part

        # code
        den_s = open("cpp_generator/mobilenetv2/Template/Function/Dense_Softmax.txt")
        den_r = open("cpp_generator/mobilenetv2/Template/Function/Dense_Relu.txt")
        dense_softmax = den_s.read()
        dense_relu = den_r.read()
        comment = ''
        comment1 = '\'\'\''

        if not use_bias:
            comment = '//'
            comment1 = '\'\'\''

        if self.config['activation'] == 'relu':  # Activation = relu
            func = dense_softmax.format(Name=self.config["name"], Input_channel=input_shape[1],
                                        Output_channel=output_shape[1], comment_begin=comment1,
                                        comment_end=comment1, comment=comment)
            self.function['code'] += func + "\n"
        else:  # Activation = softmax
            func = dense_relu.format(Name=self.config["name"], Input_channel=input_shape[1],
                                     Output_channel=output_shape[1], comment_begin=comment1,
                                     comment_end=comment1, comment=comment)
            self.function['code'] += func + "\n"


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
        ad = open("cpp_generator/mobilenetv2/Template/Function/Add.txt")
        template = ad.read()
        func = template.format(Name=self.config['name'], Input_channel1=output_shape[3], Input_width1=output_shape[1],
                               Input_height1=output_shape[2], Input_channel2=output_shape[3],
                               Input_width2=output_shape[1], Input_height2=output_shape[2],
                               Output_channel=output_shape[3], Output_width=output_shape[1],
                               Output_height=output_shape[2])
        self.function['code'] = func + "\n"
