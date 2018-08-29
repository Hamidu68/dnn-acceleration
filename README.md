# ML-acceleration  

1. Set the conditions.  
  
for example,  
```
Test_dir="Test_file/Test.csv"  
Model_name="Ex_model"  
Data_type="ap_uint<16>"  
Random_range="20"    
```  
Test_dir : path to the csv file which contains layer information  
Model_name : name of the model like vgg16, vgg19 etc.  
Data_type : data type that we use.  
Random_range : if the Random_range value is 'Num', each index of the array is initialized to a value between 1 and Num.  

2. Run ./run.sh  
```
./run.sh
```  
- 1) Run ./verifier.sh  
     ```
     ./verifier.sh $return_dir$Test_dir $Random_range  
     ```  
- 2) Create 'model_name.cpp' and 'model_name_test.cpp' files  
     ```
     python Test_cpp_Generator.py $return_dir$Test_dir $Model_name $Data_type  
     python Cpp_Generator.py $return_dir$Test_dir $Model_name $Data_type  
     ```  
     for example, 
     ```
     python Test_cpp_Generator.py ../Test-file/Test.csv vgg19 ap_uint<16> 
     python Cpp_Generator.py ../Test-file/Test.csv vgg19 ap_uint<16>
     ```
