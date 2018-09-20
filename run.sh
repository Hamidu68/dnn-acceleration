#Run Keras-verification , Generate files for Vivado
#Input : layer information( ex. test.csv )
#Output :
#1. Result of C code vs keras code
#2. model_test.cpp , model.cpp

Model_name="vgg19"
Test_dir="test_file/vgg19_test.csv"
#Test_dir="test_file/"${Model_name}".csv"
Data_type="int"
#Data_type="unsigned int"
#Data_type="float"
#Data_type="ap_unit<16>"
Random_range="10"

Use_trained_weight="T"
Trained_weight_file=${Model_name}"_weights.bin"
Image_file="image.bin"

#Run keras-verifier and compare result between c_code & keras
if [ $# -eq 0 ];then

   cd keras-verification
   
   if [ ${Use_trained_weight} -eq "T" ];then

      ./verifier.sh ../${Test_dir} ${Random_range} ${Trained_weight_file} ${Image_file}

   else

      ./verifier.sh ../${Test_dir} ${Random_range} ${Data_type}
   fi

   cd ..
fi

#Generate cpp files for Vivado
cd c-code-generation
python Test_Generator.py ../${Test_dir} ${Model_name} ${Data_type}
python Cpp_Generator.py ../${Test_dir} ${Model_name} ${Data_type}
python DAC2017_Test_Generator.py ../${Test_dir} ${Model_name} ${Data_type}
python DAC2017_Cpp_Generator.py ../${Test_dir} ${Model_name} ${Data_type}

