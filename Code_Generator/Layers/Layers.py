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