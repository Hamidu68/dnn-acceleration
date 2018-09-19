Training CNN models on tiny-imagenet
===

This contains code related to train CNN models(vgg, resnet etc) on tiny-imagenet.
Load tiny-imagenet, define the manual CNN model, train the model, and save weights of trained model.
The weights of pre-trained model is used to be loaded into HW(HLS or real FPGA).


<a name="toc"></a>
## Table of Contents

* [Load data](#load)

* [Define a CNN model](#define)

* [Train the model](#train)

* [Save weights of the trained model](#save)

* [Convert .h5 to .bin](#convert)

* [Extract csv file from the model](#extract)

<a name="load"></a>
### Load data
The tiny-imagenet dataset is in [tiny_imagenet/tiny-imagenet200](tiny_imagenet/tiny-imagenet200), and there are three folders which are train, validation, and test.

Call load_images(...) function in [tiny_imagenet/load_tiny_imagenet.py](tiny_imagenet/load_tiny_imagenet.py)
- Example usage
    ```
        X_train, Y_train, X_val, Y_val = load_images('./tiny-imagenet-200', 200)
    ```

<a name="define"></a>
### Define a CNN model
1. Make a new directory for your manual CNN model ([vgg19](vgg19)).
2. Define a function that return a keras model instance ([vgg19/vgg19.py](vgg19/vgg19.py)).
- Example usage
    ```
    from vgg19 import vgg19
        model = vgg19(weights=False,    data_format='channels_first', image_size=64, nb_classes=200)
    ```

<a name="train"></a>
### Train the model
1. Make a code for training the model ([vgg19/train.py](vgg19/train.py)).
2. Train the model
```
    $ python train.py
```
<a name="save"></a>
### Save weights of the trained model
If you see the train.py, there is a CustomCallback class.
When your model is being trained, the function in CustomCallback is called after each epochs.
The function saves the current state and weights of your model as h5 file.

<a name="convert"></a>
### Convert .h5 to .bin
For HLS, we need bin file of weights. So we need to convert h5 file to bin file.
The h5 files in [vgg19/weights_h5](vgg19/weights_h5). After converting, the bin file is saved in [vgg19/weights_bin](vgg19/weights_bin).
See the [vgg19/cvt_h5_bin.py](vgg19/cvt_h5_bin.py).
- Example usage
    ```
        $ python cvt_h5_bin.py
    ```

<a name="extract"></a>
### Extract csv file from the model
Also, you can extract configs of the model to csv file.
- Example usage
    ```
        $ python extract_config.py
    ```
