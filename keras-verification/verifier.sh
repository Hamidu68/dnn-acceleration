Test_dir=$1
Random_range=$2
Model_name=$4

Variable_dir="../Variable_Generator/"

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

Test_path="test_file/resnet50_test.csv"
python3 run.py True False False ${Test_path} ${Model_name} ${Data_type}

cd C_verifier_code/resnet50

g++ -std=c++0x C_verifier.cpp -o out
variable_path=../../Variable_Generator/
#C-code
./out ${variable_path}${Weight_file} ${variable_path}${Input_file}

#Run Verifier
<<<<<<< HEAD
g++ -std=c++0x C_Verifier.cpp -o out
./out $Variable_dir$Weight_file $Variable_dir$Bias_file $Variable_dir$Input_file
python Keras_Verifier.py $1 $Variable_dir$Weight_file $Variable_dir$Bias_file $Variable_dir$Input_file $
=======
#Keras code
cd ../../keras-verification
python Keras_Verifier.py ${Test_dir} ${Variable_dir}${Weight_file} ${Variable_dir}${Input_file} ${Data_type}

cd ../cpp_generator/
>>>>>>> jinbae

#Compare result
vimdiff Output/${Model_name}/keras_output.txt Output/${Model_name}/C_output.txt

