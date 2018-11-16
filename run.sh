#Run Keras-verification , Generate files for Vivado
#Input : layer information( ex. test.csv )
#Output :
#1. Result of C code vs keras code
#2. model_test.cpp , model.cpp

###########we need modity this file

Model_name="inceptionv3"
Test_dir="Test_file/${Model_name}_test.csv"
#Data_type="int"
#Data_type="unsigned int"
Data_type="float"
#Data_type="ap_unit<16>"
Random_range="10"

Use_trained_weight=0
Trained_weight_file=${Model_name}"_weights.bin"
Image_file="image.bin"

#Run keras-verifier and compare result between c_code & keras
if [ $# -eq 0 ];then

   cd keras-verification
   
   if [ ${Use_trained_weight} -eq 1 ];then

      ./verifier.sh ../${Test_dir} ${Random_range} ${Trained_weight_file} ${Image_file} ${Data_type} ${Model_name}

   else

      ./verifier.sh ../${Test_dir} ${Random_range} ${Data_type} ${Model_name}
   fi

   cd ..
fi

#Generate cpp files for Vivado
#cd c-code-generation
#python Test_Generator.py ../${Test_dir} ${Model_name} ${Data_type}
#python Cpp_Generator.py ../${Test_dir} ${Model_name} ${Data_type}
#python DAC2017_Test_Generator.py ../${Test_dir} ${Model_name} ${Data_type}
#python DAC2017_Cpp_Generator.py ../${Test_dir} ${Model_name} ${Data_type}

