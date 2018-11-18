##get from: https://keras.io/applications/
#from keras.applications.xception import Xception
#from keras.applications.vgg16 import VGG16
#from keras.applications.vgg19 import VGG19
from keras.applications.resnet50 import ResNet50
#from keras.applications.inception_v3 import InceptionV3
#from keras.applications.inception_resnet_v2 import InceptionResNetV2
#from keras.applications.mobilenet import MobileNet
#from keras.applications.densenet import DenseNet121
#from keras.applications.densenet import DenseNet169
#from keras.applications.densenet import DenseNet201
#from keras.applications.nasnet import NASNetLarge
#from keras.applications.nasnet import NASNetMobile
#from keras.applications.mobilenetv2 import MobileNetV2

import csv, sys, os

sys.path.insert(0, os.path.abspath(".."))
from vgg19.vgg19 import vgg19

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

def extract_configs(model,name):
    list_config = []

    #relevant_nodes
    relevant_nodes = []
    for v in model._nodes_by_depth.values():
        relevant_nodes += v
            
    for layer in model.layers:
        dict_layer = layer.get_config()
        
        dict_layer['batch_input_shape'] = layer.input_shape
        dict_layer['batch_output_shape']= layer.output_shape #add output shape for hidder layers
        dict_layer['layer_type'] = (str(layer).split()[0]).split('.')[-1]

        #custom
        connections = ''
        for node in layer._inbound_nodes:
            if relevant_nodes and node not in relevant_nodes:
                # node is not part of the current network
                continue
            for i in range(len(node.inbound_layers)):
                inbound_layer = node.inbound_layers[i].name
                if i != 0:
                    connections += '/'
                connections += inbound_layer
                
        dict_layer['connected_to'] = connections
        dict_layer['params'] = str(layer.count_params())
        
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

if __name__ == '__main__':
    #vgg19 on tiny imangenet
    model = vgg19(weights=False, data_format='channels_first', image_size=64, nb_classes=200)
    print('*'*30)
    print('Extracting csv of vgg19...')
    extract_configs(model,'vgg19')
    print('Done making csv file')
    print('*'*30)

    #resnet50 on imagenet
    model = ResNet50(include_top=True, weights='imagenet', input_tensor=None, input_shape=(224,224,3), pooling=None, classes=1000)
    print('*'*30)
    print('Extracting csv of resnet50...')
    extract_configs(model,'resnet50')
    print('Done making csv file')
    print('*'*30)