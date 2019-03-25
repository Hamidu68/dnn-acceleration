# Development of C programs for Convolutional Neural Network Accelerators

### Purpose
Creating a new toolkit for people who are using fpga.   

### Prerequisites
* python 2.7
* tensorflow
* keras(2.2)
* numpy

### System structure  
[model_info](./model_info)  
[src](./src)  
[output](./output)  
[training](./training)  
[other](./other)  

  
![structure](./other/image/structure.jpeg)
  

### Usage

#### 1. Download the repository

```
git clone "git@github.com:Hamidu68/ML-acceleration.git"
```

#### 2. Set condition

in ML-acceleration/config.json file, change the value of various keys.  
```
{
  "network": "vgg19",
  "model_info_file": "vgg19_test.csv",
  "c_code_generate": "True",
  "keras_generate": "True",
  "sw_verify": "True",
  "hw_code_generate": "False",
  "vivado_generate": "False",
  "data_type": "int",
  "random_range": "5",
  "skip_batch_layer": "False"
}
```  
* network : name of the model(network) ex. vgg19, resnet50  
* model_info_file : name of the test file which contains layer information of the model (ex. vgg19_test.csv)   
* c_code_generate : generate software code(c code of the model, {model_name}.cpp file) or not (True/False)   
* keras_generate : build model from keras or not (True/False)   
* sw_verify : verify the output value between c code and keras or not (True/False)   
* hw_code_generate : generate hardware code(optimized code, {model_name}.cpp file & {model_name}_test.cpp file)  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;or not(True/False)    
* vivado_generate : compare the value between software code and hardware code or not (True/False)   
* data_type : data type (int, unsinged int, float, ap_uint<16>)   
* random_range : number of range that will be used to generate input.bin, weight.bin value  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;(ex. 5 means input.bin, weight.bin files consist of value between 1 to 5)     
* skip_batch_layer : skip batch layer or not (True/False)   


#### 3. Run run.sh

in ML-acceleration folder, 
use the command below to run run.sh script file.   
```
./run.sh
```  
in run.sh file,  
```  
python main.py  
```  

script file 'run.sh' will run main.py file.

 ### Ongoing work
 1. code revision - structure/hareware code generator   
 2. add new models(architecture)   
 3. apply quantization/weight prunning  
