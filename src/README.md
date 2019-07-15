# src  

source code for sw test(for c vs keras), variable generator(for input, weight file), extract configs(for model info) and vivado test (optimized code generator for vivado).  

### [Model](./model)(folder)  

it saves the type of layers, templates and structure of the network that we are using.  
For more details, click the title 'Model' or [here](./Model).   

### VariableGenerator.py  

It is generated based on layer informaion(stored in csv file in ML-acceleration/model_info folder) using the command below.  


```
  python variable_generator.py ${file_path} ${Weight_file} ${Input_file} ${Random_range} ${Data_type}   
```  
- ``file_path`` : path to .csv file which contain layer information of the network(model). 
- ``Weight_file``: destination for the initialized Weight matrix (output)
- ``Input_file``: destination for the initialized Input matrix (output)
- ``Random_range``: generate the random numbers from 0 to Random_range-1
- ``Data_type``: type of the data, e.g., float


Exmple:
```
python variable_generator.py ../model_info/vgg19_test.csv init_weight.bin init_input.bin 5 float
```

In result, this code makes two .bin files. (for this example, init_weight.bin and init_input.bin files are created.) 


### sw_test.py  
Call c_generator.py and keras_generator.py and compare the result if each values of layers are same using vimdiff command. maximum error is calculated by using maximum_error.py.    
The output values of keras and C will be saved under ML-acceleration/output/model_name/output_value/.   

### c_generator.py  
Generate a c_verifire.cpp, the c code version of model.  
The output will be saved under ML-acceleration/output/model_name/.    
we run this cpp file in sw_test.py.  

### keras_generator.py 
Generate output values in Keras  
It calls keras_layer.py  

### keras_layers.py   
Contain functions of various layers which call keras function. 
for example, in add_MaxPooling2D function,  
```  
def add_MaxPooling2D(input_tensor=None, info=None, skip=False, tensors = {}):
    if skip:
        return tensors[info['connected_to']]

    # Get output tensor
    output_tensor = layers.MaxPooling2D(pool_size=eval(info['pool_size']),
                                        strides=eval(info['strides']),
                                        padding=str(info['padding']),
                                        data_format=str(info['data_format']))(input_tensor)
    return output_tensor
```  

### maximum_error.py 
Check maximum error range by comparing each value of c and keras  
and print the maximum value of the difference of each element divided by the element of keras.  

### vivado test.py
Create a optimized code for Vivado HLS.  
the code contains the comparison of the values between c code and optimized one.
it calls vivado_generator.py.    

### vivado_generator.py
Generate {model_name}.cpp and {model_name}_test.cpp.    
The output will be saved under ML-acceleration/output/model_name/.    

### extract_configs.py  
use it to generate .csv file which contains layer information of the network(model) that we are using.  
the usage of this file is explained in [ML-acceleration/model_info](../model_info) folder.  
