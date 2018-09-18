'''Train vgg19 on tiny-imagenet
'''
#system
import os, sys
import matplotlib.pyplot as plt

#keras
from keras.models import load_model
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import Adam
from keras.callbacks import Callback

#custom
from vgg19 import vgg19
sys.path.insert(0, os.path.abspath(".."))
from tiny_imagenet.load_tiny_imagenet import load_images


######Params######
model_name = 'vgg19'
data_format = 'channels_first'
image_size = 64
nb_classes = 2
nb_epochs = 30
train_batchsize = 32
val_batchsize = 4

weights_path = './weights_h5'


######Model######
cur_epoch=0
file_name = ''
for file in os.listdir(weights_path):
    file_name_list = file.split('.')
    if file_name_list[-1] == 'h5':
        cur_epoch = int(file_name_list[-2])
        file_name=file

if cur_epoch != 0:
    model = load_model(weights_path + '/' + file_name)
else:
    model = vgg19(weights=True, data_format=data_format, image_size=image_size, nb_classes=nb_classes)
    model.compile(loss='categorical_crossentropy',
              optimizer=Adam(lr=0.001),
              metrics=['acc'])
#model.summary()


######Data######
#Load images
data_path='../tiny_imagenet/tiny-imagenet-200'

X_train,Y_train,X_val,Y_val=load_images(data_path,nb_classes)
train_samples=len(X_train)
val_samples=len(X_val)

print('X_train shape:', X_train.shape)
print(X_train.shape[0], 'train samples')
print(X_val.shape[0], 'validation samples')

#Generate train data
# Change the batchsize according to your system RAM
train_datagen = ImageDataGenerator(
        rotation_range=10,
        width_shift_range=0.1,
        height_shift_range=0.1,
        horizontal_flip=True,
        fill_mode='nearest',
        data_format=data_format)
train_generator = train_datagen.flow(
        x=X_train,
        y=Y_train,
        batch_size=train_batchsize)

validation_datagen = ImageDataGenerator(data_format=data_format)
validation_generator = validation_datagen.flow(
        x=X_val,
        y=Y_val,
        batch_size=val_batchsize,
        shuffle=False)


######Logging######
class CustomCallback(Callback):
    '''
    '''
    def __init__(self, nb_epochs, model_name):
        self.nb_epochs = nb_epochs
        self.model_name = model_name
        self.f = open('./logs/' + model_name + '_log.txt', 'a')
    
    def on_batch_end(self, batch, logs={}):
        self.f.write('batch {}/{}  - loss: {:.4f}  - acc: {:.4f}\n'.format(batch+1,logs['size'],logs['loss'],logs['acc']))
        print()

    def on_epoch_begin(self, epoch, logs={}):
        self.f.write('  Epoch {}\n'.format(epoch+1))
        
    def on_epoch_end(self, epoch, logs={}):
        self.f.write('epoch {}/{}  - loss: {:.4f}  - acc: {:.4f}  - val_loss: {:.4f}  - val_acc: {:.4f}\n'.format(epoch+1,self.nb_epochs,logs['loss'],logs['acc'],logs['val_loss'],logs['val_acc']))
        model.save(weights_path + '/' + self.model_name + '_weights.' + str(epoch+1) + '.h5')
    
    def on_train_end(self, logs={}):
        self.f.close()
        print('Training End!')


######Training######
history = model.fit_generator(
      train_generator,
      #steps_per_epoch=train_samples/train_generator.batch_size ,
      epochs=nb_epochs,
      validation_data=validation_generator,
      #validation_steps=val_samples/validation_generator.batch_size,
      verbose=1,
      initial_epoch=cur_epoch,
      callbacks=[CustomCallback(nb_epochs,model_name)])


######Print result######
acc = history.history['acc']
val_acc = history.history['val_acc']
loss = history.history['loss']
val_loss = history.history['val_loss']
 
epochs = range(len(acc))

plt.figure()
plt.plot(epochs, loss, 'b', label='Training loss')
plt.plot(epochs, val_loss, 'r', label='Validation loss')
plt.title('Training and validation loss')
plt.legend()
plt.savefig('./logs/' + model_name + '_loss.JPEG')
plt.close()

plt.figure()
plt.plot(epochs, acc, 'b', label='Training acc')
plt.plot(epochs, val_acc, 'r', label='Validation acc')
plt.title('Training and validation accuracy')
plt.legend()
plt.svaefig('./logs/' + model_name + '_acc.JPEG')
plt.close()

