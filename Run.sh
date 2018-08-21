#Run Keras-verification , Generate files for Vivado
#Input : layer information( ex. test.csv )
#Output : 
#1. Result of C code vs keras code
#2. model_test.cpp , model.cpp

Test_dir="Test_file/Test.csv"
Model_name="Ex_model"
Data_type="ap_uint<16>"
Random_range="20"
return_dir="../"

cd keras-verification
./Verifier.sh $return_dir$Test_dir $Random_range

cd ../c-code-generation
python3 Test_cpp_Generator.py $return_dir$Test_dir $Model_name $Data_type
python3 Cpp_Generator.py $return_dir$Test_dir $Model_name $Data_type
