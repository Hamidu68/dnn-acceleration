# Model  
 It saves the structure of the model(network) including many types of various Layers and Templates to generate c code.   
 
#### [Layers](./Layers)    

In 'Layers' folder, various types of layers are included.  
In each file of layer, there is a class which forms c code of that layer using template file. 

#### [template](./template)  

There are 4 directories which categorize the template files.  
(Function/ Init/ Main/ Print)  
They are used to create c codes.  

#### Data.py  
It is used to save the data type of the model, data shape and name of the data.  

#### Graph.py  
It is used to specify connection between each layer of the network.  
Each layer saves layer number of input layer and its layer number.  

#### Models.py  
Based on csv file which contains layer information,  
build model using ```add_layer```, ```skip_layer ```, ```set_output```, ```get_inputs```, ```add_graph``` function.  


