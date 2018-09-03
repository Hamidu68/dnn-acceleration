
# coding: utf-8

# In[1]:


import keras
from keras.layers import Conv2D, MaxPooling2D, AveragePooling2D, BatchNormalization, ZeroPadding2D, Flatten, Dense, Activation
from keras.models import Sequential
from keras import backend as K
import numpy as np
import csv
import sys

np.set_printoptions(threshold=np.inf)
np.set_printoptions(linewidth=9223372036854775807)
csv_file=open(sys.argv[1])
csv_reader=csv.DictReader(csv_file)

weight_read=open(sys.argv[2],'rb')
bias_read=open(sys.argv[3],'rb')
input_read=open(sys.argv[4],'rb')

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
            print("[Keras_verifier.py]Calculate Conv2D"+line_str+"\n")
            filter_shape = np.asarray(row["kernel_size"][1:-1].split(", ")).astype(np.int)
            strides_shape= np.asarray(row["strides"][1:-1].split(", ")).astype(np.int)
            # Weight setting
            filters = np.empty([(int)(output_shape[3]),(int)(input_shape[3]),filter_shape[0],filter_shape[1]], dtype=np.int32)
            for m in range( (int)(output_shape[3]) ):
                for k in range( (int)(input_shape[3]) ):
                    for i in range(filter_shape[0]):
                        for j in range(filter_shape[1]):
                            filters[m][k][i][j] = np.fromfile(file=weight_read, dtype=np.int32, count=1, sep='')
            filters = np.transpose(filters,(2,3,1,0))
            # Bias setting
            bias = np.empty([(int)(output_shape[3])], dtype=np.int32)
            for m in range((int)(output_shape[3])):
                bias[m] = np.fromfile(file=bias_read, dtype=np.int32, count=1, sep='')
            #Set Convolution2D
            model.add(Conv2D(filters = (int)(output_shape[3]),kernel_size=filter_shape,strides=strides_shape,padding= row["padding"],weights=[filters,bias],activation=row["activation"],data_format='channels_first'))
            layer_name.append("Convolution2D : ")
            
        elif layer == 'MaxPooling2D' : 
            #Report Status
            print("[Keras_verifier.py]Calculate MaxPooling2D"+line_str+"\n")
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
            input_value = np.empty([1,(int)(input_shape[3]),(int)(input_shape[1]),(int)(input_shape[2])], dtype=np.int32)
            for k in range((int)(input_shape[3])):
                for i in range((int)(input_shape[1])):
                    for j in range((int)(input_shape[2])):
                        input_value[0][k][i][j] = np.fromfile(file=input_read, dtype=np.int32, count=1, sep='')
            layer_name.append("InputLayer : ")

        elif layer == 'BatchNormalization' :
            #Report Status
            print("[Keras_verifier.py]Calculate BatchNormalization"+line_str+"\n")
            #Set BatchNormalization
            model.add(BatchNormalization(axis=1,name=row["name"]))
            layer_name.append("BatchNormalization : ") 

        elif layer == 'Activation' :
            #Report Status
            print("[Keras_verifier.py]Calculate Activation(Relu)"+line_str+"\n")
            #Set activations.relu
            model.add(Activation('relu'))
            layer_name.append("Activation.Relu : ")
            
        elif layer == 'AveragePooling2D' :
            #Report Status
            print("[Keras_verifier.py]Calculate AveragePooling2D"+line_str+"\n")
            #Set strides, kernalsize 
            strides_shape= np.asarray(row["strides"][1:-1].split(", ")).astype(np.int)
            pool_shape = np.asarray(row["pool_size"][1:-1].split(", ")).astype(np.int)
            #Set AveragePooling2D
            model.add(AveragePooling2D(pool_size=pool_shape,strides=strides_shape,padding=row["padding"],data_format='channels_first'))
            layer_name.append("AveragePooling2D : ")
            
        #elif layer == 'Add' :
        elif layer =='ZeroPadding2D' :
            #Report Status
            print("[Keras_verifier.py]Calculate ZeroPadding2D"+line_str+"\n")
            #Set padding 
            padding1 = row["padding"][1:-1].split(", (")
            padding = padding1[0][1:-1].split(", ")
            #Set ZeroPadding2D
            model.add(ZeroPadding2D(padding=(int)(padding1[0]), data_format='channels_first'))
            layer_name.append("ZeroPadding2D : ")
            
        elif layer == 'Flatten' :
            #Report Status
            print("[Keras_verifier.py]Calculate Flatten"+line_str+"\n")
            #Set Flatten
            model.add(Flatten(data_format='channels_first'))
            layer_name.append("Flatten : ")
            
        elif layer == 'Dense' :
            #Report Status
            print("[Keras_verifier.py]Calculate Dense"+line_str+"\n")
            # Weight setting
            filters = np.empty([(int)(output_shape[1]),(int)(input_shape[1])], dtype=np.int32)
            for m in range((int)(output_shape[1])):
                for k in range((int)(input_shape[1])):
                    filters[m][k] = np.fromfile(file=weight_read, dtype=np.int32, count=1, sep='')
            filters = np.transpose(filters,(1,0))
            # Bias setting
            bias = np.empty([(int)(output_shape[1])], dtype=np.int32)
            for m in range((int)(output_shape[1])):
                bias[m] = np.fromfile(file=bias_read, dtype=np.int32, count=1, sep='')
            #Set Dense
            model.add(Dense(units=int(row["units"]),activation=row["activation"],weights = [filters,bias]))
            layer_name.append("Dense : ")
            
        else : 
            print("Undefined\n")
    #Report Status
    print("[Keras_verifier.py]Print Result\n")

    #Open file
    f = open('Output/keras_output.txt','w')
    
    temp = model.predict(input_value)

    #Write input_value
    f.write('{}[['.format(layer_name[0]))
    for k in range(input_value.shape[1]):
        f.write('[')
        for x in range(input_value.shape[2]):
            f.write('[')
            for y in range(input_value.shape[3]):
                f.write('{} '.format(int(input_value[0][k][x][y])))
            if x != (input_value.shape[2] - 1):
                f.write(']\n   ')
            else:
                f.write(']')
        if k != (input_value.shape[1] - 1):
            f.write(']\n\n  ')
    f.write(']]]\n\n')

    #Write each layers output
    before_output = input_value
    for i in range(line_num):
        get_3rd_layer_output = K.function([model.layers[i].input], [model.layers[i].output])
        layer_output = get_3rd_layer_output([before_output])[0]
        before_output = layer_output

        if len(layer_output.shape) == 4:
            f.write('{}[['.format(layer_name[i+1]))
            for k in range(layer_output.shape[1]):
                f.write('[')
                for x in range(layer_output.shape[2]):
                    f.write('[')
                    for y in range(layer_output.shape[3]):
                        f.write('{} '.format(int(layer_output[0][k][x][y])))
                    if x != (layer_output.shape[2] - 1):
                        f.write(']\n   ')
                    else:
                        f.write(']')
                if k != (layer_output.shape[1] - 1):
                    f.write(']\n\n  ')
            f.write(']]]\n\n')
            
        elif len(layer_output.shape) == 2:
            f.write('{}[['.format(layer_name[i+1]))
            for k in range(layer_output.shape[1]):
                f.write('{} '.format(int(layer_output[0][k])))
            f.write(']]\n\n')
            
    #model.summary()
    f.close()


