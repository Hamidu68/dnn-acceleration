# src  

contain functions for c and keras generator, verifier, variable generator(for input, weight file) and optimized one.  

### [Model](./Model)(folder)  

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


### sw_generator.py  
Create a c file of the network that contains the c code of each layer and building model.  
The output c file is saved in ML-acceleration/output folder.  

### keras_generator.py 
Calculate output values of each layer by Keras  
The output values are saved in ML-acceleration/output/{model_name}/output_value folder.  

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

### print_keras.py  
It is used in 'keras_generator.py' file.  
Create keras output file to compare the values with c file and calculate maximum error.   

### maximum_error.py 
Check maximum error range by comparing each value of c and keras  
and print the maximum value of the difference of each element divided by the element of keras.  

### sw_verifier.py  
Compare the values between generated c file and keras using vimdiff command.  
And maximum error is calculated by using maximum_error.py.  

### hw_generator.py (not completed)    
Create a c file of the network that contains the optimized version of c code of each layer and building model.  

### vivado_generator.py (not completed)  
Code that contains the comparison of the values between generated c code and optimized version.    


### extract_configs.py  
use it to generate .csv file which contains layer information of the network(model) that we are using.  
the usage of this file is explained in [ML-acceleration/model_info](../model_info) folder.  
