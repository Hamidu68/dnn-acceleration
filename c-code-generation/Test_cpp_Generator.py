
# coding: utf-8

# In[10]:


import csv
import sys
from string import Template
    
# sys.argv[1] = Test.csv
# sys.argv[2] = model name
# sys.argv[3] = Data_type

csv_file=open(sys.argv[1])
csv_reader=csv.DictReader(csv_file)

if __name__ == "__main__" :    
    
    static_v =""
    initialization = ""
    line_count = -1
    model_name = sys.argv[2]
    global input_shape
    global output_shape
    def_model = "("
    func_model = "("
    #####Load template#####
    # Open file
    m = open("../Template/Main/Test_cpp.txt")
    i_conv = open("../Template/Init/Conv_var_Initializer_int.txt")
    i_dense = open("../Template/Init/Dense_var_Initializer_int.txt")
    i_input = open("../Template/Init/Input_var_Initializer_int.txt")
    main = Template(m.read())
    Init_conv = Template(i_conv.read())
    Init_dense = Template(i_dense.read())
    Init_input = Template(i_input.read())
    
    #####Generate Function depending on layer_type #####
    for row in csv_reader:
        #Count Line number
        line_count+= 1;
        line_num = str(line_count)
        # layer_type = convolution2D(I,O,B,W) (padding option: valid , same)
        if row["layer_type"] == 'Conv2D' :
            #Get Input, Output shape
            input_shape = row["batch_input_shape"][1 : -1].split(", ")
            output_shape = row["batch_output_shape"][1 : -1].split(", ")
            filter_shape = row["kernel_size"][1:-1].split(", ")
            #Static_variables(W,B)
            static_v += "static DATA_T W"+line_num+"[" + output_shape[3] + "][" + input_shape[3] + "][" + filter_shape[0] + "][" + filter_shape[1] + "];\n\t";
            static_v += "static DATA_T B"+line_num + "[" + output_shape[3] + "];\n\t";
            m = {'Input_channel' : input_shape[3],'Output_channel' : output_shape[3],'Filter_width' : filter_shape[0],'Filter_height' : filter_shape[1],'line_number' : line_count}
            #Initialization
            initialization += Init_conv.substitute(m) + "\n\t"
            #def model
            def_model +=",DATA_T W"+line_num+"[" + output_shape[3] + "][" + input_shape[3] + "][" + filter_shape[0] + "][" + filter_shape[1] + "], DATA_T B"+line_num + "[" + output_shape[3] + "]"
            #func_model
            func_model +=",W"+line_num + ",B"+line_num
            
        # layer_type = InputLayer
        elif row["layer_type"] =="InputLayer":
            #Get Input shape
            input_shape = row["batch_input_shape"][1 : -1].split(", ")
            #static_variables (I)
            static_v += "static DATA_T I[" + input_shape[3] + "][" + input_shape[1] + "][" + input_shape[2] + "];\n\t";
            static_v += "static DATA_T O0_SW[" + input_shape[3] + "][" + input_shape[1] + "][" + input_shape[2] + "];\n\t";
            m = {'Input_channel':input_shape[3],'Input_width':input_shape[1],'Input_height':input_shape[2]}
            #Initialization
            initialization += Init_input.substitute(m) + "\n\t"
            #def model
            def_model +="DATA_T I[" + input_shape[3] + "][" + input_shape[1] + "][" + input_shape[2] + "]"
            #func_model
            func_model +="I"
            
        # layer_type = Dense(I,W,B,O) (Activation option : relu , softmax)
        elif row["layer_type"] == "Dense":
            #Get Input, Output shape
            input_shape = row["batch_input_shape"][1 : -1].split(", ")
            output_shape = row["batch_output_shape"][1 : -1].split(", ")
            #static_variable (W,B)
            static_v += "static DATA_T W"+line_num+ "[" + output_shape[1] + "][" + input_shape[1] + "];\n\t";
            static_v += "static DATA_T B"+line_num+ "[" + output_shape[1] + "];\n\t";
            m = {'Input_channel' : input_shape[1],'Output_channel' : output_shape[1],'line_number' : line_count}
            #Initialization
            initialization += Init_dense.substitute(m) + "\n\t"
            #def model
            def_model +=",DATA_T W"+line_num+"[" + output_shape[1] + "][" + input_shape[1] + "], DATA_T B"+line_num + "[" + output_shape[1] + "]"
            #func_model
            func_model +=",W"+line_num + ",B"+line_num
    
    # Output Shape 3D
    if len(output_shape) == 4 :
        #SW_static_variables (O_SW)
        static_v += "static DATA_T O_SW[" + output_shape[3] + "][" + output_shape[1] + "][" + output_shape[2] + "];\n\t"; 
        #HW_static_variables (O_HW)
        static_v += "static DATA_T O_HW[" + output_shape[3] + "][" + output_shape[1] + "][" + output_shape[2] + "];\n\t"; 
        def_model += ", DATA_T O[" + output_shape[3] + "][" + output_shape[1] + "][" + output_shape[2] + "]);\n"  
        model_function =  model_name + "_top" + func_model + ",O_HW);\n"
        model_function += "  " + model_name + "_sw" + func_model + """,O_SW);\n
   int err_cnt = 0;
   for (m=0; m<"""+output_shape[3]+"""; m++) {
       for (x=0; x<"""+output_shape[1]+"""; x++) {
          for (y=0; y<"""+output_shape[2]+"""; y++) {
              if (O_HW[m][x][y] != O_SW[m][x][y]) {
                printf("SW: O[%d][%d][%d] = %d", m, x, y, O_SW[m][x][y]);
                printf("HW: O[%d][%d][%d] = %d", m, x, y, O_HW[m][x][y]);
                err_cnt++;}
           }
       }
   }\n"""
    #Output Shape 1D   
    else :
        #SW_static_variables (O_SW)
        static_v += "static DATA_T O_SW[" + output_shape[1] + "];\n\t"; 
        #HW_static_variables (O_HW)
        static_v += "static DATA_T O_HW[" + output_shape[1] + "];\n\t"; 
        def_model += ", DATA_T O[" + output_shape[1] + "]);\n"   
        model_function =  model_name + "_top" + func_model + ",O_HW);\n"
        model_function += "  " + model_name + "_sw" + func_model + """,O_SW);\n
    int err_cnt = 0;
    for (m=0; m<"""+output_shape[1]+"""; m++) {
        if (O_HW[m] != O_SW[m]) {
            printf("SW: O[%d] = %d", m, O_SW[m]);
            printf("HW: O[%d] = %d", m, O_HW[m]);
            err_cnt++;}
    }\n"""
    
    model_definition = "void " + model_name + "_top" + def_model
    model_definition += "void " + model_name + "_sw" + def_model
    
    # Generate CPP file
    f = {'def_model':model_definition,'static_variables':static_v, 'Initialization':initialization, 'model_function' : model_function, 'D_type' : sys.argv[3]}    
    c_file = main.substitute(f) + "\n";
    file = open("Output/"+model_name + "_test.cpp",'w')
    file.write(c_file)
    file.close() 

