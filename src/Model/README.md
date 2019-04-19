# Model  
 It saves the structure of the model(network) including many types of various Layers and Templates to generate c code.   
 
#### [Layers](./Layers)    

In 'Layers' folder, various types of layers are included.  
In each file, there are classes which form c code and optimized code using template. 

#### [template](./template)  

There are 4 directories which categorize the template files.  
(function/ init/ main/ print)  
They are used to create codes.  

#### Models.py  
Build model based on csv file(layer information).  
build model using ```add_layer```, ```skip_layer ```, ```set_output```, ```get_inputs```, ```add_graph``` function.  


