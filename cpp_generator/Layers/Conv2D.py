from .Data import Data
from .Layers import Layers

class Conv2D(Layers):

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

        #params
        self.function['params'] = self.get_params()

        #code
        self.function['code'] = self.template.format(name='',
                                                     post=self.post,
                                                     params='',
                                                     inner='')
