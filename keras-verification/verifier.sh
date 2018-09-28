#test_dir="Test-file/Test.csv"

Input_file="init_Input.bin"
Weight_file="init_Weight.bin"
Bias_file="init_Bias.bin"

Variable_dir="../Variable_Generator/"
Result_dir="vimdiff.txt"
return_dir="../"

#Generate Variable
#Input: layer info / Output : c_verifier.cpp
cd $Variable_dir
g++ -std=c++0x Variable_Generator.cpp -o out
./out $1 $Weight_file $Bias_file $Input_file $2

#Generate C_Verifier
cd ../keras-verification
python C_Verifier_Generator.py $1 $3

#Run Verifier
g++ -std=c++0x C_Verifier.cpp -o out
./out $Variable_dir$Weight_file $Variable_dir$Bias_file $Variable_dir$Input_file
python Keras_Verifier.py $1 $Variable_dir$Weight_file $Variable_dir$Bias_file $Variable_dir$Input_file $

#Compare result
vimdiff Output/keras_output.txt Output/C_output.txt

