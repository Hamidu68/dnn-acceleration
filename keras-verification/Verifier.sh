#test_dir="Test-file/Test.csv"

Input_file="init_Input.txt"
Weight_file="init_Weight.txt"
Bias_file="init_Bias.txt"

Verification_dir="../keras-verification/"
Result_dir="vimdiff.txt"
return_dir="../"
#Generate Variable 
#Input: layer info / Output : c_verifier.cpp
cd ../Variable_Generator
g++ Variable_Generator.cpp -o out
./out $return_dir$1 $Verification_dir$Weight_file $Verification_dir$Bias_file $Verification_dir$Input_file

#Generate C_Verifier
cd ../keras-verification
python3 C_Verifier_Generator.py $return_dir$1

#Run Verifier
python3 Keras_Verifier.py $return_dir$1 $Weight_file $Bias_file $Input_file
g++ C_Verifier.cpp -o out
./out $Weight_file $Bias_file $Input_file

#Compare result
vimdiff Output/keras_output.txt Output/C_output.txt

