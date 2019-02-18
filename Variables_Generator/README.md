Random Generator for Input, Weight value.  
It is generated based on layer informaion(csv) using the command below.  


```
  python variable_generator.py ${Test_dir} ${Weight_file} ${Input_file} ${Random_range} ${Data_type}   
```  
- ``Test_dir`` : path to .csv file. 
- ``Weight_file``: destination for the initialized Weight matrix (output)
- ``Input_file``: destination for the initialized Input matrix (output)
- ``Random_range``: generate the random numbers from 0 to Random_range-1
- ``Data_type``: type of the data, e.g., float


Exmple:
```
python variable_generator.py ../Test_file/vgg19_test.csv init_weight.bin init_input.bin 5 float
```

In result, this code makes two .bin files.
