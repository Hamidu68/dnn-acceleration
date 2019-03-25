# model_info

Store .csv files which contain layer information of each model(network).   
For file name, normally we use {model_name}_test.csv form (for example, vgg16_test.csv)  

To generate these files,  

1. Import the model(network) that you want to generate in ML-acceleration/src/extract_configs.py file  

for example, if we want to generate resnet50 file,  
```
from keras.applications.resnet50 import ResNet50  
```  
write these at the top of extract_configs.py.  

2. Build model and call extract_configs function   
For example, if we want to generate resnet50_test.csv file in model_info folder,  
put absolute path or relative path including file name in {file_path} without file extension.    
(ex. {file_path} : '../model_info/resnet50_test')    
After that, build model and call extract_configs function like the below.   
```
model = ResNet50(include_top=True, weights='imagenet', input_tensor=None, input_shape=(224,224,3), pooling=None, classes=1000)
print('**********************************resnet50***********************************')
extract_configs(model, {file_path})
```  
  
3. Run extract_configs.py file by using the command below.  

```
python extract_configs.py 
``` 




