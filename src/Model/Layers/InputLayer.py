from .Data import Data
from .Layers import Layers


class InputLayer(Layers):
    def __init__(self, config={}, inputs=[], dtype='DATA_T', layer_odr=0, post=''):
        super(InputLayer,self).__init__(config, inputs, dtype, layer_odr, post)
        self.inputs.append(Data(dtype=self.dtype, shape=eval(config['batch_input_shape']), name='I'))
        self.output = self.inputs[0]
