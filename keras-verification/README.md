# This verifier generates a keras model with the same structure as the generated c code and compares the values.

1. Run ./Verifier.sh Test_file/Test.csv (your test layer information)
2. Layer information : Test.csv
3. Generate Variable(Input,Weight,Bias) : Variable_Generator.cpp ( you can use python code also )
4. Generate C_Verifier : C_Verifier_Generator.py
5. Run Verifier with same Input, Weight and Bias : C_verifier.c / keras_verifier.py
6. Compare result : c_output.txt / keras_output.txt
7. You can see compared result using vimDiff
