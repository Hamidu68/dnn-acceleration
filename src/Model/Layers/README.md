# Layers  

### Class is defined in each Layer  

In Layers.py file, class 'Layers' is defined.   

```
class Layers(object):
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

```

In each python file(for example, ```Conv2D.py```), class of each layer is defined which inherits features from Layers class(base class).    

![Layers](./other/image/Layers.jpg)

And in each class, c code of each layer is created using template file.  
