
# coding: utf-8

# In[9]:


import numpy as np
import csv
import random
import sys

# sys.argv[1] = Test.csv (layer file) 
# sys.argv[2] = Init_weight.txt (Output-weight file)
# sys.argv[3] = Init_Bias.txt (Output-weight file)
# sys.argv[4] = Init_Input.txt (Output-weight file)
if __name__ == "__main__":
    #open File
    Weight = open(sys.argv[2],'w')
    Bias = open(sys.argv[3],'w')
    Input = open(sys.argv[4],'w')
    
    #open csv
    csv_file = open(sys.argv[1])
    csv_reader = csv.DictReader(csv_file)
    
    weight_file = ""
    bias_file =""
    input_file =""
    
    for row in csv_reader:
        #Get Input, Output shape
        input_shape = row["batch_input_shape"][1 : -1].split(", ")
        output_shape = row["batch_output_shape"][1 : -1].split(", ")
        
        w_size = 0
        b_size = 0
        i_size = 0
        
        # Find out number of variable that have to be initialized
        if row["layer_type"] == 'Conv2D' :
            filter_shape = np.asarray(row["kernel_size"][1:-1].split(", ")).astype(np.int)
            w_size = int(input_shape[3])*int(output_shape[3])*filter_shape[0]*filter_shape[1]
            b_size = int(output_shape[3])
        elif row["layer_type"] =="InputLayer":
            i_size = (int)(input_shape[3])*(int)(input_shape[2])*(int)(input_shape[1])      
        elif row["layer_type"] == "Dense":
            w_size = int(input_shape[1])*int(output_shape[1])
            b_size = int(output_shape[1])            
        else :
            continue;
            
        # Make Random varible
        for i in range (w_size) :
            temp = random.randrange(1,10)
            weight_file = weight_file + " " + str(temp)
        if w_size !=0 : weight_file += "\n"

        for i in range (b_size) :
            temp = random.randrange(1,10)
            bias_file = bias_file + " " + str(temp)
        if b_size !=0 :bias_file += "\n"
        
        for i in range (i_size) :
            temp = random.randrange(1,20)
            input_file = input_file + " " + str(temp)
        if i_size !=0 :input_file += "\n"
    
    #Write file
    Weight.write(weight_file)
    Bias.write(bias_file)
    Input.write(input_file)
    
    #Close file
    Weight.close()
    Bias.close()
    Input.close()

