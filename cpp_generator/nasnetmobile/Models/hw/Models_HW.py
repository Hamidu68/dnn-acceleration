from ...Layers import *
from ..Models import Models


class Models_HW(Models):
    def __init(self, model_name='', dtype='DATA_T', post='_HW'):
        super().__init__(model_name, dtype, post)

    def add_layer(self, config={}):
        layer_name = config['name']
        layer_type = config['layer_type']

        # Switch
        if layer_type == 'Conv2D':
            self.add_graph(layer_name, config['connected_to'])
            inputs = self.get_inputs(layer_name)
            self.layers.append(Conv2D_HW(config, inputs, dtype=self.dtype, layer_odr=self.layer_num, post=self.post))
            
        elif layer_type == 'MaxPooling2D':
            self.add_graph(layer_name, config['connected_to'])
            inputs = self.get_inputs(layer_name)
            self.layers.append(MaxPooling2D_HW(config, inputs, dtype=self.dtype, layer_odr=self.layer_num, post=self.post))
            
        else:
            super().add_layer(config)
            return
        
        self.layer_num += 1
        return
