Test_dir=$1
Random_range=$2
Model_name=$4

Variable_dir="../variable_generator/"

Weight_file="init_weight.bin"
Input_file="init_input.bin"


if [ $# -eq 4 ] ; then

   #Generate Variable
   #Input: layer info / Output : c_verifier.cpp
   Data_type='int'


   cd ${Variable_dir}

   python variable_generator.py ${Test_dir} ${Weight_file} ${Input_file} ${Random_range} ${Data_type}
   cd ..

elif [ $# -eq 6 ] ; then

   Weight_file=$3
   Input_file=$4
   Data_type=$5
fi

#Generate C_Verifier

Test_path="Test_file/${Model_name}_test.csv"
python3 run.py True False ${Test_path} ${Model_name} ${Data_type}

cd C_verifier_code/${Model_name}

g++ -std=c++0x C_verifier.cpp -o out
variable_path=../../variable_generator/
#C-code
./out ${variable_path}${Weight_file} ${variable_path}${Input_file}

#Run Verifier
#Keras code
cd ../../keras-verification
python Keras_Verifier.py ${Test_dir} ${Variable_dir}${Weight_file} ${Variable_dir}${Input_file} ${Data_type} ${Model_name}

cd ../cpp_generator/${Model_name}

#Compare result
vimdiff Output/keras_output.txt Output/C_output.txt

