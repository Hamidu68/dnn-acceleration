import sys
import os
import numpy as np
from PIL import Image
from keras.utils import np_utils

def get_annotations_map(path):
    '''
    '''
    valAnnotationsPath = path + '/val/val_annotations.txt'
    valAnnotationsFile = open(valAnnotationsPath, 'r')
    valAnnotationsContents = valAnnotationsFile.read()
    valAnnotations = {}

    for line in valAnnotationsContents.splitlines():
    	pieces = line.strip().split()
    	valAnnotations[pieces[0]] = pieces[1]

    return valAnnotations


def load_images(path,nb_classes):
    #Load images
    
    print('Loading ' + str(nb_classes) + ' classes')

    X_train=np.zeros([nb_classes*500,3,64,64],dtype='uint8')
    Y_train=np.zeros([nb_classes*500], dtype='uint8')

    trainPath=path+'/train'

    print('loading training images...');

    i=0
    j=0
    annotations={}
    for sChild in os.listdir(trainPath):
        sChildPath = os.path.join(os.path.join(trainPath,sChild),'images')
        annotations[sChild]=j
        for c in os.listdir(sChildPath):
            X=np.array(Image.open(os.path.join(sChildPath,c)))
            if len(np.shape(X))==2:
                X_train[i]=np.array([X,X,X])
            else:
                X_train[i]=np.transpose(X,(2,0,1))
            Y_train[i]=j
            i+=1
        j+=1
        if (j >= nb_classes):
            break

    #X_train = X_train.astype(np.float32)
    # convert class vectors to binary class matrices
    Y_train = np_utils.to_categorical(Y_train, nb_classes)
    
    print('finished loading training images('+str(i)+')')

    val_annotations_map = get_annotations_map(path)

    X_val = np.zeros([nb_classes*50,3,64,64],dtype='uint8')
    Y_val = np.zeros([nb_classes*50], dtype='uint8')


    print('loading validation images...')

    i = 0
    valPath=path+'/val/images'
    for sChild in os.listdir(valPath):
        if val_annotations_map[sChild] in annotations.keys():
            sChildPath = os.path.join(valPath, sChild)
            X=np.array(Image.open(sChildPath))
            if len(np.shape(X))==2:
                X_val[i]=np.array([X,X,X])
            else:
                X_val[i]=np.transpose(X,(2,0,1))
            Y_val[i]=annotations[val_annotations_map[sChild]]
            i+=1
        else:
            pass

    #X_val = X_val.astype(np.float32)
    # convert class vectors to binary class matrices
    Y_val = np_utils.to_categorical(Y_val, nb_classes)
    
    print('finished loading validation images('+str(i)+')')

    return X_train,Y_train,X_val,Y_val


def load_test_images(path,nb_classes):
    #Load images
    
    trainPath=path+'/train'

    print('loading validation images...');

    j=0
    annotations={}
    for sChild in os.listdir(trainPath):
        sChildPath = os.path.join(os.path.join(trainPath,sChild),'images')
        annotations[sChild]=j
        j+=1
        if (j >= nb_classes):
            break

    val_annotations_map = get_annotations_map(path)

    X_val = np.zeros([nb_classes*50,3,64,64],dtype='uint8')
    Y_val = np.zeros([nb_classes*50], dtype='uint8')

    i = 0
    valPath=path+'/val/images'
    for sChild in os.listdir(valPath):
        if val_annotations_map[sChild] in annotations.keys():
            sChildPath = os.path.join(valPath, sChild)
            X=np.array(Image.open(sChildPath))
            if len(np.shape(X))==2:
                X_val[i]=np.array([X,X,X])
            else:
                X_val[i]=np.transpose(X,(2,0,1))
            Y_val[i]=annotations[val_annotations_map[sChild]]
            i+=1
        else:
            pass

    #X_val = X_val.astype(np.float32)
    # convert class vectors to binary class matrices
    Y_val = np_utils.to_categorical(Y_val, nb_classes)
    
    print('finished loading validation images('+str(i)+')')


    print('loading test images...');

    X_test = np.array([])

    i=0
    testPath=path+'/test/images'
    for sChild in os.listdir(testPath):
        sChildPath = os.path.join(testPath, sChild)
        X=np.array(Image.open(sChildPath))
        if len(np.shape(X))==2:
            X_test[i]=np.array([X,X,X])
        else:
            X_test[i]=np.transpose(X,(2,0,1))
        i+=1
    
    #X_test = X_test.astype(np.float32)

    print('finished loading test images('+str(i)+')')

    return X_val,Y_val,X_test


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    path='./tiny-imagenet-200'
    X_train,Y_train,X_val,Y_val = load_images(path,2)
    

    fig1 = plt.figure()
    fig1.suptitle('Train data')
    ax1 = fig1.add_subplot(221)
    ax1.axis("off")
    ax1.imshow(np.transpose(X_train[0],(1,2,0)))
    ax1.set_title(np.argmax(Y_train[0]))
    ax2 = fig1.add_subplot(222)
    ax2.axis("off")
    ax2.imshow(np.transpose(X_train[499],(1,2,0)))
    ax2.set_title(np.argmax(Y_train[499]))
    ax3 = fig1.add_subplot(223)
    ax3.axis("off")
    ax3.imshow(np.transpose(X_train[500],(1,2,0)))
    ax3.set_title(np.argmax(Y_train[500]))
    ax4 = fig1.add_subplot(224)
    ax4.axis("off")
    ax4.imshow(np.transpose(X_train[999],(1,2,0)))
    ax4.set_title(np.argmax(Y_train[999]))

    plt.show()

    fig2 = plt.figure()
    fig2.suptitle('Validation data')
    ax1 = fig2.add_subplot(221)
    ax1.axis("off")
    ax1.imshow(np.transpose(X_val[0],(1,2,0)))
    ax1.set_title(np.argmax(Y_val[0]))
    ax2 = fig2.add_subplot(222)
    ax2.axis("off")
    ax2.imshow(np.transpose(X_val[49],(1,2,0)))
    ax2.set_title(np.argmax(Y_val[49]))
    ax3 = fig2.add_subplot(223)
    ax3.axis("off")
    ax3.imshow(np.transpose(X_val[50],(1,2,0)))
    ax3.set_title(np.argmax(Y_val[50]))
    ax4 = fig2.add_subplot(224)
    ax4.axis("off")
    ax4.imshow(np.transpose(X_val[99],(1,2,0)))
    ax4.set_title(np.argmax(Y_val[99]))
    
    plt.show()
