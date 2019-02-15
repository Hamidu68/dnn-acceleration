# ML-acceleration  

The way to generate software code of various models

1. Set the conditions in run.sh file. 
  
for example,  
```
Model_name="vgg19"
Test_dir="Test_file/${Model_name}_test.csv"
Data_type="float"
Random_range="5"  
```  
Model_name : name of the model like vgg16, vgg19 etc.  
Test_dir : path to the csv file which contains layer information  
Data_type : data type that we use.  
Random_range : if the Random_range value is 'Num', each index of the array is initialized to a value between 1 and Num.  

2. Change __init__.py file.

In ML-acceleration/cpp_generator folder, there are various folders that contain different layers, output, template in different models.
To import one of these folders, change __init__.py file like this.

```  
from .{Model_name} import *
```   

for example,  
```  
from .vgg19 import *
``` 


3. Run ./run.sh  
```
./run.sh
```  
  In run.sh, it contains the command below.
  
  ```
  ./verifier.sh ../${Test_dir} ${Random_range} ${Data_type} ${Model_name}
  ```
  each argument is already explained above.      
   
