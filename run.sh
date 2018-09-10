#Run Keras-verification , Generate files for Vivado
#Input : layer information( ex. test.csv )
#Output : 
#1. Result of C code vs keras code
#2. model_test.cpp , model.cpp

Test_dir="Test_file/vgg19_test.csv"
Model_name="vgg19"
Data_type="ap_uint<16>"
Random_range="20"
return_dir="../"

cd keras-verification
./verifier.sh $return_dir$Test_dir $Random_range

cd ../c-code-generation
python Test_cpp_Generator.py $return_dir$Test_dir $Model_name $Data_type
python Test_cpp_DAC2017_Generator.py $return_dir$Test_dir $Model_name $Data_type
python Cpp_DAC2017_Generator.py $return_dir$Test_dir $Model_name $Data_type
python Cpp_Generator.py $return_dir$Test_dir $Model_name $Data_type

