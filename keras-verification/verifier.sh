Test_dir=$1
Random_range=$2

Variable_dir="../variable_generator/"

Weight_file="init_weights.bin"
Input_file="init_inputs.bin"

if [ &# -eq 3 ];then

   #Generate Variable
   #Input: layer info / Output : c_verifier.cpp
   Data_type=$3
   
   cd ${Variable_dir}
   python variable_generator.py ${Test_dir} ${Weight_file} ${Input_file} ${Random_range} ${Data_type}
   cd ..

elif [ &# -eq 4 ];then

   Weight_file=$3
   Input_file=$4

fi

#Generate C_Verifier
cd keras-verification
python C_Verifier_Generator.py ${Test_dir}
g++ -std=c++0x C_Verifier.cpp -o out

#Run Verifier
#Keras code
python Keras_Verifier.py ${Test_dir} ${Variable_dir}${Weight_file} ${Variable_dir}${Input_file} ${Data_type}
#C-code
./out ${Variable_dir}${Weight_file} ${Variable_dir}${Input_file}

#Compare result
vimdiff Output/keras_output.txt Output/C_output.txt

