#test_dir="Test-file/Test.csv"

Input_file="init_Input.txt"
Weight_file="init_Weight.txt"
Bias_file="init_Bias.txt"

Variable_dir="../Variable_Generator/"
Result_dir="vimdiff.txt"
return_dir="../"

#Generate Variable 
#Input: layer info / Output : c_verifier.cpp
cd $Variable_dir
g++ Variable_Generator.cpp -o out
./out $1 $Weight_file $Bias_file $Input_file $2

#Generate C_Verifier
cd ../keras-verification
python3 C_Verifier_Generator.py $1

#Run Verifier
python3 Keras_Verifier.py $1 $Variable_dir$Weight_file $Variable_dir$Bias_file $Variable_dir$Input_file
g++ C_Verifier.cpp -o out
./out $Variable_dir$Weight_file $Variable_dir$Bias_file $Variable_dir$Input_file

#Compare result
vimdiff Output/keras_output.txt Output/C_output.txt

