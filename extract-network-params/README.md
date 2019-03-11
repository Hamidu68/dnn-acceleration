###extract configs of Keras networks.  

first, import the model from Keras that you are using.  
```
from keras.applications.{Model_name} import {Model_name(in capital letters)}  
```
for example,   
```
from keras.applications.vgg16 import VGG16  
```

second, set the parameters that you want to extract.  

for example, 
 ```
params = [
    "name",
    "layer_type",
    "batch_input_shape",
    "batch_output_shape",
    'connected_to',
    'params',
    "filters",
    "kernel_size",
    "activation",
    "padding",
    "strides",
    "pool_size",
    "units",
    ]
```  

third, call function extrac_configs.  

for example, 

```
model = VGG16   (include_top=True, weights='imagenet', input_tensor=None, input_shape=(224,224,3), pooling=None, classes=1000)
print('**********************************vgg16**************************************')
extract_configs(model,'vgg16')
```  

Lastly, run extract_configs.py file.  

```
python extract_configs.py 
```

network: vgg16, vgg19, resnet50, ...
