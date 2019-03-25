# output  
### Save produced code after running run.sh (for now, it generates software code)  
 
   
&nbsp;&nbsp;&nbsp;&nbsp;By running run.py, the c code of that network is created in the folder named model that we are using.  

In each folder,  
```
C_verifier.cpp file  : c code of the network that we are using
output_value folder : store the output of c code and keras based on randomly generated weight and input values.  
```
is created.  

( for example, if we use vgg16 model, by running run.py and passing parameters, C_verifier.cpp file and output_value folder are created in vgg16 folder.)  
