# model_info

&nbsp;&nbsp;&nbsp;&nbsp;Store .csv files which contain layer information of each model(network).   
For file name, normally we use {model_name}_test.csv form (for example, vgg16_test.csv)  

&nbsp;&nbsp;&nbsp;&nbsp;To generate these files,  

#### 1. Import the model(network) that you want to generate in ML-acceleration/src/extract_configs.py file  

&nbsp;&nbsp;&nbsp;&nbsp;For example, if we want to generate resnet50.csv file to extract layer information,  
```
from keras.applications.resnet50 import ResNet50  
```  
write these at the top of extract_configs.py.  
  
#### 2. Build model and call extract_configs function   
&nbsp;&nbsp;&nbsp;&nbsp;For example, if we want to generate resnet50_test.csv file in model_info folder,  
put absolute path or relative path including file name in {file_path} without file extension.    
(ex. {file_path} : '../model_info/resnet50_test')    
&nbsp;&nbsp;&nbsp;&nbsp;After that, build model and call extract_configs function like the below.   
```
model = ResNet50(include_top=True, weights='imagenet', input_tensor=None, input_shape=(224,224,3), pooling=None, classes=1000)
print('**********************************resnet50***********************************')
extract_configs(model, {file_path})
```  
  
#### 3. Run extract_configs.py file by using the command below.  

```
python extract_configs.py 
```  
In this case, resnet50_test.csv file is created in ML-acceleration/model_info folder.  




