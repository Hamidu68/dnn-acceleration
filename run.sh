#Run Keras-verification , Generate files for Vivado
#Input : layer information( ex. test.csv )
#Output :
#1. Result of C code vs keras code
#2. model_test.cpp , model.cpp

Test_dir="Test_file/vgg19_test.csv"
Model_name="vgg19"
Data_type="int"
Random_range="20"
return_dir="../"
Relu="False"

cd keras-verification
./verifier.sh $return_dir$Test_dir $Random_range $Relu

cd ../c-code-generation
python Test_Generator.py $return_dir$Test_dir $Model_name $Data_type
python DAC2017_Test_Generator.py $return_dir$Test_dir $Model_name $Data_type
python DAC2017_Cpp_Generator.py $return_dir$Test_dir $Model_name $Data_type $Relu
python Cpp_Generator.py $return_dir$Test_dir $Model_name $Data_type $Relu

