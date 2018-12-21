Random Generator for Input, Weight value.  
It is generated based on layer informaion(csv) using the command below.  


```
  python variable_generator.py ${input csv directory} ${output directory} ${model name} ${output Weight matrix} ${output Input matrix} ${Random_range} ${Data_type}   
```  
- ``Test_dir`` : path to .csv file. 
- ``Weight_file``: destination for the initialized Weight matrix (output)
- ``Input_file``: destination for the initialized Input matrix (output)
- ``Random_range``: generate the random numbers from 0 to Random_range-1
- ``{Data_type}``: type of the data, e.g., float


Exmple:
```
python variable_generator.py ../Test_file/ ../C_verifier_code/ vgg16 wight input 5 float
```

In result, this code makes two .bin files.
