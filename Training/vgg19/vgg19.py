'''written by Jinbae Park 
'''

#from keras.applications import VGG19
from keras import Model
from keras import layers
from keras import utils
from keras.regularizers import l2

def load_top(weights=False, model=None):
    #Load pre-trained weights on ImageNet for convolution layers(not dense).
    if weights == True:
        #file path
        WEIGHTS_PATH_NO_TOP = ('https://github.com/fchollet/deep-learning-models/'
                       'releases/download/v0.1/'
                       'vgg19_weights_tf_dim_ordering_tf_kernels_notop.h5')
        weights_path = utils.get_file(
                'vgg19_weights_tf_dim_ordering_tf_kernels_notop.h5',
                WEIGHTS_PATH_NO_TOP,
                cache_subdir='models',
                file_hash='253f8cb515780f3b799900260a226db6')
        model.load_weights(weights_path)
    elif weights is not None:
        model.load_weights(weights)

    return model

def vgg19(weights=True,
    data_format='channels_first',
    image_size=64,
    nb_classes=200):
    
    """Instantiates the VGG19 architecture.

    Optionally loads weights pre-trained on ImageNet.
    Note that the data format convention used by the model is
    the one specified in your Keras config at `~/.keras/keras.json`.

    # Arguments
        weights: one of `None` (random initialization),
              'imagenet' (pre-training on ImageNet),
              or the path to the weights file to be loaded.
        data_format: data_format has to be 'channels_first'
            (with input_shape=(3,size,size)
            or 'channels_last' (with input_shape=(size,size,3)
        image_size: width and height of image
        nb_classes: optional number of classes to classify images
            into.

    # Returns
        A Keras model instance.
    """
    
    #Input image
    if data_format == 'channels_last':
        inputs = layers.Input(shape=(image_size,image_size,3))
    elif data_format == 'channels_first':
        inputs = layers.Input(shape=(3,image_size,image_size))

    #Block 1
    x = layers.Conv2D(64, (3, 3),
                      activation='relu',
                      padding='same',
                      kernel_regularizer=l2(0.01),
                      data_format=data_format,
                      name='block1_conv1')(inputs)
    x = layers.Conv2D(64, (3, 3),
                      activation='relu',
                      padding='same',
                      kernel_regularizer=l2(0.01),
                      data_format=data_format,
                      name='block1_conv2')(x)
    x = layers.MaxPooling2D((2, 2), strides=(2, 2), data_format=data_format, name='block1_pool')(x)

    #Block 2
    x = layers.Conv2D(128, (3, 3),
                      activation='relu',
                      padding='same',
                      kernel_regularizer=l2(0.01),
                      data_format=data_format,
                      name='block2_conv1')(x)
    x = layers.Conv2D(128, (3, 3),
                      activation='relu',
                      padding='same',
                      kernel_regularizer=l2(0.01),
                      data_format=data_format,
                      name='block2_conv2')(x)
    x = layers.MaxPooling2D((2, 2), strides=(2, 2), data_format=data_format, name='block2_pool')(x)
    
    #Block 3
    x = layers.Conv2D(256, (3, 3),
                      activation='relu',
                      padding='same',
                      kernel_regularizer=l2(0.01),
                      data_format=data_format,
                      name='block3_conv1')(x)
    x = layers.Conv2D(256, (3, 3),
                      activation='relu',
                      padding='same',
                      kernel_regularizer=l2(0.01),
                      data_format=data_format,
                      name='block3_conv2')(x)
    x = layers.Conv2D(256, (3, 3),
                      activation='relu',
                      padding='same',
                      kernel_regularizer=l2(0.01),
                      data_format=data_format,
                      name='block3_conv3')(x)
    x = layers.Conv2D(256, (3, 3),
                      activation='relu',
                      padding='same',
                      kernel_regularizer=l2(0.01),
                      data_format=data_format,
                      name='block3_conv4')(x)
    x = layers.MaxPooling2D((2, 2), strides=(2, 2), data_format=data_format, name='block3_pool')(x)

    #Block 4
    x = layers.Conv2D(512, (3, 3),
                      activation='relu',
                      padding='same',
                      kernel_regularizer=l2(0.01),
                      data_format=data_format,
                      name='block4_conv1')(x)
    x = layers.Conv2D(512, (3, 3),
                      activation='relu',
                      padding='same',
                      kernel_regularizer=l2(0.01),
                      data_format=data_format,
                      name='block4_conv2')(x)
    x = layers.Conv2D(512, (3, 3),
                      activation='relu',
                      padding='same',
                      kernel_regularizer=l2(0.01),
                      data_format=data_format,
                      name='block4_conv3')(x)
    x = layers.Conv2D(512, (3, 3),
                      activation='relu',
                      padding='same',
                      kernel_regularizer=l2(0.01),
                      data_format=data_format,
                      name='block4_conv4')(x)
    x = layers.MaxPooling2D((2, 2), strides=(2, 2), data_format=data_format, name='block4_pool')(x)

    #Block 5
    x = layers.Conv2D(512, (3, 3),
                      activation='relu',
                      padding='same',
                      kernel_regularizer=l2(0.01),
                      data_format=data_format,
                      name='block5_conv1')(x)
    x = layers.Conv2D(512, (3, 3),
                      activation='relu',
                      padding='same',
                      kernel_regularizer=l2(0.01),
                      data_format=data_format,
                      name='block5_conv2')(x)
    x = layers.Conv2D(512, (3, 3),
                      activation='relu',
                      padding='same',
                      kernel_regularizer=l2(0.01),
                      data_format=data_format,
                      name='block5_conv3')(x)
    x = layers.Conv2D(512, (3, 3),
                      activation='relu',
                      padding='same',
                      kernel_regularizer=l2(0.01),
                      data_format=data_format,
                      name='block5_conv4')(x)
    x = layers.MaxPooling2D((2, 2), strides=(2, 2), data_format=data_format, name='block5_pool')(x)

    #Load weights of top
    if weights != False:
        x = load_top(weights=weights, model=Model(inputs, x, name='vgg19_top')).output

    #Add classification Block
    x = layers.Flatten(data_format=data_format)(x)
    x = layers.Dense(2048, activation='relu')(x)
    x = layers.Dropout(0.5)(x)
    x = layers.Dense(2048, activation='relu')(x)
    x = layers.Dropout(0.5)(x)
    x = layers.Dense(nb_classes, activation='softmax')(x)

    #Create model
    model = Model(inputs, x, name='vgg19')
    
    return model


if __name__ == '__main__':
    model = vgg19(weights=True, data_format='channels_first', image_size=64, nb_classes=200)
    print(model)
