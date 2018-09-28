from ..Layers import *

class Models():
    def __init__(self, model_name='', dtype='DATA_T', post=''):
        self.model_name = model_name
        self.dtype=dtype
        self.post = post
        self.layer_num = 0
        self.layers = []
        self.inputs = []
        self.outputs = []
        #self.weights = []
        self.graphs = {}

    def add_graph(self, name='', connected=''):
        self.graphs[name] = {'in':[],
                             'odr':self.layer_num}
        for pre_name in connected.split('/'):
            self.graphs[name]['in'].append(self.graphs[pre_name]['odr'])

    def get_inputs(self, name=''):
        inputs=[]
        for pre_odr in self.graphs[name]['in']:
            inputs.append(self.layers[pre_odr].output)
        return inputs
    
    def set_output(self):
        if layer_num != -1:
            data = self.layers[-1].output
            data.name = 'O' + self.post
            self.outputs.append(data)

    def skip_layer(self, config={}):
        self.graphs[config['name']] = self.graphs[config['connected_to']]

    def add_layer(self, config={}):
        layer_name = config['name']
        layer_type = config['layer_type']
        
        #Switch
        if layer_type == 'InputLayer':
            layer = InputLayer(config, dtype=self.dtype, layer_odr=self.layer_num, post=self.post)
            self.layers.append(layer)
            self.inputs.append(layer.inputs[0])
            self.graphs[layer_name] = {'in':[-1],
                                       'odr':self.layer_num}
        elif layer_type == 'Conv2D':
            self.add_graph(layer_name, config['connected_to'])
            inputs = self.get_inputs(layer_name)
            self.layers.append(Conv2D(config, inputs, dtype=self.dtype, layer_odr=self.layer_num, post=self.post))
            
        elif layer_type == 'MaxPooling2D':
            self.add_graph(layer_name, config['connected_to'])
            inputs = self.get_inputs(layer_name)
            self.layers.append(MaxPooling2D(config, inputs, dtype=self.dtype, layer_odr=self.layer_num, post=self.post))
            
        elif layer_type == 'BatchNormalization':
            self.add_graph(layer_name, config['connected_to'])
            inputs = self.get_inputs(layer_name)
            self.layers.append(BatchNormalization(config, inputs, dtype=self.dtype, layer_odr=self.layer_num, post=self.post))
            
        elif layer_type == 'Activation':
            self.add_graph(layer_name, config['connected_to'])
            inputs = self.get_inputs(layer_name)
            self.layers.append(Activation(config, inputs, dtype=self.dtype, layer_odr=self.layer_num, post=self.post))
            
        elif layer_type == 'AveragePooling2D':
            self.add_graph(layer_name, config['connected_to'])
            inputs = self.get_inputs(layer_name)
            self.layers.append(AveragePooling2D(config, inputs, dtype=self.dtype, layer_odr=self.layer_num, post=self.post))
            
        elif layer_type == 'ZeroPadding2D':
            self.add_graph(layer_name, config['connected_to'])
            inputs = self.get_inputs(layer_name)
            self.layers.append(ZeroPadding2D(config, inputs, dtype=self.dtype, layer_odr=self.layer_num, post=self.post))
            
        elif layer_type == 'Flatten':
            self.add_graph(layer_name, config['connected_to'])
            inputs = self.get_inputs(layer_name)
            self.layers.append(Flatten(config, inputs, dtype=self.dtype, layer_odr=self.layer_num, post=self.post))
            
        elif layer_type == 'Dense':
            self.add_graph(layer_name, config['connected_to'])
            inputs = self.get_inputs(layer_name)
            self.layers.append(Dense(config, inputs, dtype=self.dtype, layer_odr=self.layer_num, post=self.post))
            
        elif layer_type == 'Add':
            self.add_graph(layer_name, config['connected_to'])
            inputs = self.get_inputs(layer_name)
            self.layers.append(Add(config, inputs, dtype=self.dtype, layer_odr=self.layer_num, post=self.post))
            
        else:
            print('Undefined Layer: {}'.format(layer_type))
            self.layer_num -= 1
        
        self.layer_num += 1
        return
