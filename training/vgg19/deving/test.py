#system
import os, sys
import numpy as np
import matplotlib.pyplot as plt

#keras
from keras.models import load_model
from keras.preprocessing.image import ImageDataGenerator

#custom
sys.path.insert(0, os.path.abspath(".."))
from tiny_imagenet.load_tiny_imagenet import load_test_images



def test_images(h5_path = './weights_h5',
                model_name = 'vgg19',
                epoch = 0,
                data_format = 'channels_first',
                nb_classes=200):
    '''
    '''
    #params
    val_batchsize = 32
    test_batchsize = 32
    
    #load the model
    if epoch != 0:
        model = load_model(h5_path + '/' + model_name + '_weights.' + str(epoch) + '.h5')
    else:
        file_name=''
        for file in os.listdir(h5_path):
            file_name_list = file.split('.')
            if file_name_list[-1] == 'h5':
                file_name=file
        model = load_model(h5_path + '/' + file_name)


    #Load images
    data_path='../tiny_imagenet/tiny-imagenet-200'

    X_val,Y_val,X_test=load_test_images(data_path,nb_classes)
    val_samples=len(X_val)
    test_samples=len(X_test)

    print('X_val shape:', X_val.shape)
    print(X_val.shape[0], 'validation samples')
    print('X_test shape:', X_test.shape)
    print(X_test.shape[0], 'test samples')


    # Create a generator for validation prediction
    validation_datagen = ImageDataGenerator(data_format=data_format)
    validation_generator = validation_datagen.flow(
            x=X_val,
            y=Y_val,
            batch_size=val_batchsize,
            shuffle=False)
 
    # Get the ground truth from generator
    ground_truth = validation_generator.classes
 
    # Get the label to class mapping from the generator
    label2index = validation_generator.class_indices
    # Getting the mapping from class index to class label
    idx2label = dict((v,k) for k,v in label2index.items())
 
    # Get the predictions from the model using the generator
    predictions = model.predict_generator(
            validation_generator,
            #steps=validation_generator.samples/validation_generator.batch_size,
            verbose=1)
    predicted_classes = np.argmax(predictions,axis=1)
 
    errors = np.where(predicted_classes != ground_truth)[0]
    print("No. of validation errors = {}/{}".format(len(errors),val_samples))
 
    # Show the errors
    for i in range(len(errors)):
        pred_class = np.argmax(predictions[errors[i]])
        pred_label = idx2label[pred_class]
     
        title = 'Original label:{}, Prediction :{}, confidence : {:.3f}'.format(
            fnames[errors[i]].split('/')[0],
            pred_label,
            predictions[errors[i]][pred_class])
     
        original = load_img('{}/{}'.format(validation_dir,fnames[errors[i]]))
        plt.figure(figsize=[7,7])
        plt.axis('off')
        plt.title(title)
        plt.imshow(original)
        plt.show()


    # Create a generator for test prediction
    test_datagen = ImageDataGenerator(data_format=data_format)
    test_generator = test_datagen.flow(
            x=X_test,
            batch_size=test_batchsize,
            shuffle=False)


if __name__ == '__main__':
    test_images(nb_classes=2)
