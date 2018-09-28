from .Data import Data

class Layers():
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
    template = '''
    void {name}{post}({params}) {
        {inner}
        return;
    }
    '''
    def __init__(self, config={}, inputs=[], dtype='DATA_T', layer_odr=0, post=''):
        super().__init__(config, inputs, dtype, layer_odr, post)

        #get shape
        input_shape = eval(self.config['batch_input_shape'])
        output_shape = eval(self.config['batch_output_shape'])
        kernel_size = eval(self.config['kernel_size'])
        strides = eval(self.config['strides'])
        dilation_rate = eval(self.config['dilation_rate'])
        padding = str(self.config['padding'])
        use_bias = eval(self.config['use_bias'])
        
        #set_output
        self.set_output(output_shape[1:], self.layer_odr)

        #set_weight
        weight_shape=((output_shape[3],input_shape[3],kernel_size[0],kernel_size[1]))
        self.weights.append(Data(dtype=self.dtype, shape=weight_shape, name='W{}'.format(self.layer_odr)))
        if use_bias == True:
            self.weights.append(Data(dtype=self.dtype, shape=(output_shape[3],), name='B{}'.format(self.layer_odr)))

        #init part
        self.weights_odr.append([3,2,0,1])  #keras: kernel_width, kernel_height, input_channel, output_channel
        if use_bias == True:
            self.weights_odr.append([0])

        #code
        self.function['code'] = self.template.format(name='',
                                                     post=self.post,
                                                     params='',
                                                     inner='')


class Conv2D_DAC2017(Layers):
    template = '''
    void {name}{post}({params}) {
        {inner}
        return;
    }
    '''
    def __init__(self, config={}, inputs=[], dtype='DATA_T', layer_odr=0, post=''):
        super().__init__(config, inputs, dtype, layer_odr, post)

        #get shape
        input_shape = eval(self.config['batch_input_shape'])
        output_shape = eval(self.config['batch_output_shape'])
        kernel_size = eval(self.config['kernel_size'])
        strides = eval(self.config['strides'])
        dilation_rate = eval(self.config['dilation_rate'])
        padding = str(self.config['padding'])
        use_bias = eval(self.config['use_bias'])
        
        #set_output
        self.set_output(output_shape[1:], self.layer_odr)

        #set_weight
        weight_shape=((output_shape[3],input_shape[3],kernel_size[0],kernel_size[1]))
        self.weights.append(Data(dtype=self.dtype, shape=weight_shape, name='W{}'.format(self.layer_odr)))
        if use_bias == True:
            self.weights.append(Data(dtype=self.dtype, shape=(output_shape[3],), name='B{}'.format(self.layer_odr)))

        #init part
        self.weights_odr.append([3,2,0,1])  #keras: kernel_width, kernel_height, input_channel, output_channel
        if use_bias == True:
            self.weights_odr.append([0])

        #code
        

        
class MaxPooling2D(Layers):
    template = '''
    void {name}{post}({params}) {
        {inner}
        return;
    }
    '''
    def __init__(self, config={}, inputs=[], dtype='DATA_T', layer_odr=0, post=''):
        super().__init__(config, inputs, dtype, layer_odr, post)

        #get shape

        #set_output

        #set_weight

        #init part

        #code


class MaxPooling2D_HW(Layers):
    template = '''
    void {name}{post}({params}) {
        {inner}
        return;
    }
    '''
    def __init__(self, config={}, inputs=[], dtype='DATA_T', layer_odr=0, post=''):
        super().__init__(config, inputs, dtype, layer_odr, post)

        #get shape

        #set_output

        #set_weight

        #init part

        #code

        
class BatchNormalization(Layers):
    template = '''
    void {name}{post}({params}) {
        {inner}
        return;
    }
    '''
    def __init__(self, config={}, inputs=[], dtype='DATA_T', layer_odr=0, post=''):
        super().__init__(config, inputs, dtype, layer_odr, post)

        #get shape

        #set_output

        #set_weight

        #init part

        #code

        
class Activation(Layers):
    template = '''
    void {name}{post}({params}) {
        {inner}
        return;
    }
    '''
    def __init__(self, config={}, inputs=[], dtype='DATA_T', layer_odr=0, post=''):
        super().__init__(config, inputs, dtype, layer_odr, post)

        #get shape

        #set_output

        #set_weight

        #init part

        #code

        
class AveragePooling2D(Layers):
    template = '''
    void {name}{post}({params}) {
        {inner}
        return;
    }
    '''
    def __init__(self, config={}, inputs=[], dtype='DATA_T', layer_odr=0, post=''):
        super().__init__(config, inputs, dtype, layer_odr, post)

        #get shape

        #set_output

        #set_weight

        #init part

        #code

        
class ZeroPadding2D(Layers):
    template = '''
    void {name}{post}({params}) {
        {inner}
        return;
    }
    '''
    def __init__(self, config={}, inputs=[], dtype='DATA_T', layer_odr=0, post=''):
        super().__init__(config, inputs, dtype, layer_odr, post)

        #get shape

        #set_output

        #set_weight

        #init part

        #code

        
class Flatten(Layers):
    template = '''
    void {name}{post}({params}) {
        {inner}
        return;
    }
    '''
    def __init__(self, config={}, inputs=[], dtype='DATA_T', layer_odr=0, post=''):
        super().__init__(config, inputs, dtype, layer_odr, post)

        #get shape

        #set_output

        #set_weight

        #init part

        #code

        
class Dense(Layers):
    template = '''
    void {name}{post}({params}) {
        {inner}
        return;
    }
    '''
    def __init__(self, config={}, inputs=[], dtype='DATA_T', layer_odr=0, post=''):
        super().__init__(config, inputs, dtype, layer_odr, post)

        #get shape

        #set_output

        #set_weight

        #init part

        #code

        
class Add(Layers):
    template = '''
    void {name}{post}({params}) {
        {inner}
        return;
    }
    '''
    def __init__(self, config={}, inputs=[], dtype='DATA_T', layer_odr=0, post=''):
        super().__init__(config, inputs, dtype, layer_odr, post)


        #get shape

        #set_output

        #set_weight

        #init part

        #code

        
