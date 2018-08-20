# This verifier generates a keras model with the same structure as the generated c code and compares the values.
# Run ./Verifier.sh Test_file/Test.csv (your test layer information)
# Layer information : Test.csv
# Generate Variable(Input,Weight,Bias) : Variable_Generator.cpp ( you can use python code also )
# Generate C_Verifier : C_Verifier_Generator.py
# Run Verifier with same Input, Weight and Bias : C_verifier.c / keras_verifier.py
# compared result : c_output.txt / keras_output.txt
# You can see compared result using vimDiff
# Generate by JiyoungAn
