# This verifier generates a keras model with the same structure as the generated c code and compares the values.

1. Run ./Verifier.sh Test_file/Test.csv (your test layer information)  
```
./verifier.sh $return_dir$Test_dir $Random_range
```  
$return_dir$Test_dir : path to Test.csv file which contains layer information of the model   
$Random_range : set a numerical range of variables to be initialized (ex. 20 means 1 to 20)  
  
  
2. Layer information : Test.csv  
3. Generate Variable(Input,Weight,Bias)  
```  
g++ Variable_Generator.cpp -o out  
```  
or you can use python code also (Variable_Generator.py)  

4. Generate C_Verifier : C_Verifier_Generator.py is created with the command below
```
python C_Verifier_Generator.py  
```  
5. Run Verifier with same Input, Weight and Bias : C_verifier.c / keras_verifier.py  
```
g++ C_Verifier.cpp -o out  
./out $path to Weight_file $path to Bias_file $path to Input_file
```
6. Compare result : c_output.txt / keras_output.txt  

7. You can see compared result using vimDiff  
