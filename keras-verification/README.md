# This verifier generates a keras model with the same structure as the generated c code and compares the values.

1. Run ./verifier.sh test_file/test.csv (your test layer information)  
```
./verifier.sh ../${Test_dir} ${Random_range}
```  
../${Test_dir} : path to test.csv file which contains layer information of the model   
${Random_range} : set a numerical range of variables to be initialized (ex. 20 means 1 to 20)    
  
2. Layer information : test.csv  

3. Generate Variable(Inputs,Weights)  
```  
  python variable_generator.py ${Test_dir} ${Weight_file} ${Input_file} ${Random_range} ${Data_type}  
```  

4. Generate C_Verifier : C_Verifier_Generator.py is created with the command below
```
python C_Verifier_Generator.py  
```  

5. Run Verifier with same Inputs, Weights : C_verifier.c / keras_verifier.py  
```
g++ C_Verifier.cpp -o out  
./out ${path_to_Weight_file} ${path_to_Input_file}
```  
```
python Keras_Verifier.py $1 ${path_to_Weight_file} ${path_to_Input_file}  
```  

6. Compare result : c_output.txt / keras_output.txt  

7. You can see compared result using vimDiff  
