
# coding: utf-8

# In[1]:


import keras
from keras.layers import Conv2D, MaxPooling2D, AveragePooling2D, BatchNormalization, ZeroPadding2D, Flatten, Dense, Activation
from keras.models import Sequential
from keras import backend as K
import numpy as np
import csv
import sys

csv_file=open(sys.argv[1])
csv_reader=csv.DictReader(csv_file)

weight_read=open(sys.argv[2],'r')
bias_read=open(sys.argv[3],'r')
input_read=open(sys.argv[4],'r')

global input_value

global model

global result_file

line_num=-1
layer_name=[]

if __name__ == "__main__":
    result_file =""
    model = Sequential()
    for row in csv_reader :
        line_num=line_num+1
        line_str = str(line_num)
        layer=row["layer_type"]
        #Load value(input, output, filter)
        input_shape =  row["batch_input_shape"][1 : -1].split(", ")
        output_shape = row["batch_output_shape"][1 : -1].split(", ")

        if layer =='Conv2D' :
            #Report Status
            print("[Keras_verifier.py]Calcluate Conv2D"+line_str+"\n")
            filter_shape = np.asarray(row["kernel_size"][1:-1].split(", ")).astype(np.int)
            strides_shape= np.asarray(row["strides"][1:-1].split(", ")).astype(np.int)
            # Weight setting
            line = weight_read.readline()
            line = line.split(' ')
            weight=np.asarray(line).astype(np.int)
            filters=np.reshape(weight,((int)(output_shape[3]),(int)(input_shape[3]),filter_shape[0],filter_shape[1])) 
            filters = np.transpose(filters,(2,3,1,0))
            # Bias setting
            line = bias_read.readline()
            line = line.split(' ')
            bias=np.asarray(line).astype(np.int)
            #Set Convolution2D
            model.add(Conv2D(filters = (int)(output_shape[3]),kernel_size=filter_shape,strides=strides_shape,padding= row["padding"],weights=[filters,bias],activation=row["activation"],data_format='channels_first'))
            layer_name.append("Convolution2D : ")
            
        elif layer == 'MaxPooling2D' : 
            #Report Status
            print("[Keras_verifier.py]Calcluate MaxPooling2D"+line_str+"\n")
            #Set strides, kernalsize 
            strides_shape= np.asarray(row["strides"][1:-1].split(", ")).astype(np.int)
            pool_shape = np.asarray(row["pool_size"][1:-1].split(", ")).astype(np.int)
            #Set MaxPooling2D
            model.add(MaxPooling2D(pool_size=pool_shape,strides=strides_shape,padding=str(row["padding"]),data_format='channels_first'))
            layer_name.append("MaxPooling2D : ")

        elif layer == 'InputLayer' :
            #Report Status
            print("[Keras_verifier.py]InputLayer\n")
            #Set layer_value as input value
            line = input_read.readline()
            line = line.split(' ')
            input_value=np.asarray(line).astype(np.int) 
            input_value=np.reshape(input_value,(1,(int)(input_shape[3]),(int)(input_shape[1]),(int)(input_shape[2])))
            layer_name.append("InputLayer : ")

        elif layer == 'BatchNormalization' :
            #Report Status
            print("[Keras_verifier.py]Calcluate BatchNormalization"+line_str+"\n")
            #Set BatchNormalization
            model.add(BatchNormalization(axis=1,name=row["name"]))
            layer_name.append("BatchNormalization : ") 

        elif layer == 'Activation' :
            #Report Status
            print("[Keras_verifier.py]Calcluate Activation(Relu)"+line_str+"\n")
            #Set activations.relu
            model.add(Activation('relu'))
            layer_name.append("Activation.Relu : ")
            
        elif layer == 'AveragePooling2D' :
            #Report Status
            print("[Keras_verifier.py]Calcluate AveragePooling2D"+line_str+"\n")
            #Set strides, kernalsize 
            strides_shape= np.asarray(row["strides"][1:-1].split(", ")).astype(np.int)
            pool_shape = np.asarray(row["pool_size"][1:-1].split(", ")).astype(np.int)
            #Set AveragePooling2D
            model.add(AveragePooling2D(pool_size=pool_shape,strides=strides_shape,padding=row["padding"],data_format='channels_first'))
            layer_name.append("AveragePooling2D : ")
            
        #elif layer == 'Add' :
        elif layer =='ZeroPadding2D' :
            #Report Status
            print("[Keras_verifier.py]Calcluate ZeroPadding2D"+line_str+"\n")
            #Set padding 
            padding1 = row["padding"][1:-1].split(", (")
            padding = padding1[0][1:-1].split(", ")
            #Set ZeroPadding2D
            model.add(ZeroPadding2D(padding=(int)(padding1[0]), data_format='channels_first'))
            layer_name.append("ZeroPadding2D : ")
            
        elif layer == 'Flatten' :
            #Report Status
            print("[Keras_verifier.py]Calcluate Flatten"+line_str+"\n")
            #Set Flatten
            model.add(Flatten(data_format='channels_first'))
            layer_name.append("Flatten : ")
            
        elif layer == 'Dense' :
            #Report Status
            print("[Keras_verifier.py]Calcluate Dense"+line_str+"\n")
            # Weight setting
            line = weight_read.readline()
            line = line.split(' ')
            weight=np.asarray(line).astype(np.int)
            filters=np.reshape(weight,((int)(output_shape[1]),(int)(input_shape[1])))
            filters = np.transpose(filters,(1,0))
            # Bias setting
            line = bias_read.readline()
            line = line.split(' ')
            bias=np.asarray(line).astype(np.int)
            #Set Dense
            model.add(Dense(units=int(row["units"]),activation=row["activation"],weights = [filters,bias]))
            layer_name.append("Dense : ")
            
        else : 
            print("Undefined\n")
    #Report Status
    print("[Keras_verifier.py]Print Result\n")
    temp = model.predict(input_value)
    result_file = result_file + layer_name[0] + str(np.asarray(input_value).astype(np.int)) + "\n\n"
    for i in range(line_num):
        get_3rd_layer_output = K.function([model.layers[0].input], [model.layers[i].output])
        layer_output = get_3rd_layer_output([input_value])[0]
        result_file = result_file + layer_name[i+1] + str(np.asarray(layer_output).astype(np.int)) + "\n\n" 
    #model.summary()
    f = open('Output/keras_output.txt','w')
    f.write(result_file)
    f.close()


