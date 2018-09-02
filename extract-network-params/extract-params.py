##get from: https://keras.io/applications/
# from keras.applications.xception import Xception
from keras.applications.vgg16 import VGG16
from keras.applications.vgg19 import VGG19
from keras.applications.resnet50 import ResNet50
# from keras.applications.inception_v3 import InceptionV3
# from keras.applications.inception_resnet_v2 import InceptionResNetV2
# from keras.applications.mobilenet import MobileNet
# from keras.applications.densenet import DenseNet121
# from keras.applications.densenet import DenseNet169
# from keras.applications.densenet import DenseNet201
# from keras.applications.nasnet import NASNetLarge
# from keras.applications.nasnet import NASNetMobile
# from keras.applications.mobilenetv2 import MobileNetV2

import csv, sys

params = [
    "name",
    "layer_type",
    "batch_input_shape",
    "batch_output_shape",
    "filters",
    "kernel_size",
    "activation",
    "padding",
    "strides",
    "pool_size",
    "units",
    ]

def extract_configs(model,name):
    list_config = []
    for layer in model.layers:
        dict_layer = layer.get_config()
        print (layer.name)
     #   exit()
        dict_layer['batch_input_shape'] = layer.input_shape
        dict_layer['batch_output_shape']= layer.output_shape #add output shape for hidder layers
        dict_layer['layer_type'] = (str(layer).split()[0]).split('.')[-1]
        list_config.append(dict_layer)

    keys=[]
    for layer in model.layers: #get union of keys (input layer has different keys)
        keys=list(set().union(keys,layer.get_config().keys()))
    
    #add new params to keys! (new params that are not exists in summary of Keras)
    for param in params:
        if param not in keys:
            keys.append(param)

    #sort parmaters to be write in the csv file
    #params.reverse()        
    for param in params[::-1]:
        keys.insert(0, keys.pop(keys.index(param)))

    with open(name+'.csv', 'wb') as output_file:
        dict_writer = csv.DictWriter(output_file, keys)
        dict_writer.writeheader()
        dict_writer.writerows(list_config)


model = VGG16   (include_top=True, weights='imagenet', input_tensor=None, input_shape=(224,224,3), pooling=None, classes=1000)
print('**********************************vgg16**************************************')
extract_configs(model,'vgg16')



model = VGG19   (include_top=True, weights='imagenet', input_tensor=None, input_shape=(224,224,3), pooling=None, classes=1000)
print('**********************************vgg19**************************************')
extract_configs(model,'vgg19')


model = ResNet50(include_top=True, weights='imagenet', input_tensor=None, input_shape=(224,224,3), pooling=None, classes=1000)
print('**********************************resnet50***********************************')
extract_configs(model,'resent50')
