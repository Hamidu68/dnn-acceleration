# This verifier compares the values between keras and generated software code.

verifier.sh file is automatically called by run.sh file. 

This file contains some command below. 

1. Generate Inputs, Weights values  
```
python variable_generator.py ${Test_dir} ${Weight_file} ${Input_file} ${Random_range} ${Data_type}
``` 

${Test_dir} : path to test.csv file which contains layer information of the model (ex. ../Test_file/vgg19_test.csv)
${Weight_file} : path to init_weight.bin file to generate (ex. ../cpp_generator/vgg19/init_weight.bin)
${Input_file} : path to init_input.bin file to generate (ex. ../cpp_generator/vgg19/init_input.bin)
${Random_range} : set a numerical range of variables to be initialized (ex. 20 means 1 to 20) 
${Data_type} : data type (ex. float)

2. Generate C code

```
python3 run.py True False ${Test_path} ${Model_name} ${Data_type}
```

argv[1] : whether to contain software code (boolean value) (ex. True if you want to contain sw code)
argv[2] : whether to contain hardware code (boolean value) (ex. False if you don't want to contain hw code)
argv[3] : path to test.csv file like "Test_file/${Model_name}_test.csv" (ex. Test_file/vgg19_test.csv)
argv[4] : data type (ex. float)


3. Run C_Verifier with same Inputs, Weights values : C_verifier.c / keras_verifier.py  

in C_verifier_code/{Model_name} folder, 
```
g++ C_Verifier.cpp -o out  
./out ${1} ${2}
```  
${1} : path to init_weight file (ex. ../../variable_generator/init_weight.bin)
${2} : path to init_input file (ex. ../../variable_generator/init_input.bin)

```
python Keras_Verifier.py ${Test_dir} ${2} ${3} ${Data_type} ${Model_name}
  
```  
${Test_dir} : path to test.csv file (ex. ../Test_file/vgg19_test.csv)
${2} : path to init_weight file (ex. ../../variable_generator/init_weight.bin)
${3} : path to init_input file (ex. ../../variable_generator/init_input.bin)
${Data_type} : data type(ex. float)
${Model_name} : name of the model (ex. vgg19)


4. Compare result : c_output.txt / keras_output.txt  

```
vimdiff Output/keras_output.txt Output/C_output.txt
```

5. You can see compared result using vimDiff  
