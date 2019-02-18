# Development of C programs for Convolutional Neural Network Accelerators



### Prerequisites
* python 2.7
* tensorflow
* keras(2.2)
* numpy


  

### Usage

#### Run run.py

in ML-acceleration folder,
```
python run.py <SW_test> <HW_test> <DAC2017_test> <Test_file> <model_name> <data_type>
```
SW_test : generate software code or not(True/False)  
HW_test : generate hardware code or not(True/False)  
DAC2017_test : generate DAC2017 code or not(True/False)  
Test_file : name of the test file (ex. vgg19_test.csv)  
model_name : name of the model(network) ex. vgg19  
data_type : data type (int, unsinged int, float, ap_uint<16>)  

  
  
