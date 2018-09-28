# ML-acceleration

Create model_name.cpp (ex. vgg19.cpp) by using the command below.  

```  
python Cpp_Generator.py $layer_csvfile_path $Model_name $Data_type  
```  
  
Create model_name_test.cpp (ex. vgg19_test.cpp) by using the command below.  
  
```  
python Test_cpp_Generator.py $layer_csvfile_path $Model_name $Data_type  
``` 

$layer_csvfile_path : a relative path to csv file which contains layer information of model. (ex. ../Test_file/Test.csv)  
$Model name : ex) vgg16, vgg19, ResNet50 etc.    
$Data_type : for setting data type that we use. (ex. ap_uint<16>)  
