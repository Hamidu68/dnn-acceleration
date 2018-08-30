
# coding: utf-8

# Import
import csv
import sys
from string import Template

#sys.argv[1]=Test.csv
#sys.argv[2]=model name
#sys.argv[3]=Data type

# Main 1) Define variable

if __name__ == "__main__":
    SW_def_func =""
    HW_def_func =""
    SW_functions = ""
    HW_functions = ""
    SW_static_v = ""
    HW_static_v = ""
    Static_variables="" #Argument to Cpp.txt
    top_func_argument="" #Argument to top.txt
    Output_variables="" #Argument to sw.txt
    SW_variables="" #Argument to sw.txt
    assign_value="" #Argument to top.txt
    Optimized_code="" #Argument to top_func.txt
    variables="" #Argument to top_func.txt
    Stream_declaration="" #Argument to Cpp.txt
    line_count = -1
    conv_count=0
    pool_count = 0
    global input_shape
    global output_shape
    global first_input_shape    

    # Main 2) Load Template

    # Open file
    batch_normal = open("../Template/Function/BatchNormalization.txt")
    conv_s = open("../Template/Function/Conv2D_same.txt")
    conv_v = open("../Template/Function/Conv2D_valid.txt")
    ad = open("../Template/Function/Add.txt")
    den_s = open("../Template/Function/Dense_Softmax.txt")
    den_r = open("../Template/Function/Dense_Relu.txt")
    fla = open("../Template/Function/Flatten.txt")
    gap = open("../Template/Function/GlobalAveragePooling.txt")
    gmp = open("../Template/Function/GlobalMaxPooling.txt")
    mxp = open("../Template/Function/MaxPooling2D.txt")
    avp = open("../Template/Function/AveragePooling2D.txt")
    rl = open("../Template/Function/Relu.txt")
    sm = open("../Template/Function/Softmax.txt")
    zp = open("../Template/Function/ZeroPadding.txt")
    m = open("../Template/Main/Cpp.txt")
    st=open("../Template/Function/Stream_io.txt")
    st_d=open("../Template/Function/Stream_io_dense.txt")
    t_func=open("../Template/Function/top_func.txt")
    t=open("../Template/Function/top.txt")
    t_func_dense=open("../Template/Function/top_func_dense.txt")
    t_dense=open("../Template/Function/top_dense.txt")
    
    sw=open("../Template/Function/sw.txt")

    #Open HW file
    hw_conv_s = open("../Template/Function/Conv2D_same_HW.txt")
    hw_conv_v = open("../Template/Function/Conv2D_valid_HW.txt")
    hw_maxp = open("../Template/Function/MaxPooling2D_HW.txt")
    # Read Template
    BatchNormalization = Template(batch_normal.read())
    Conv2D_same = Template(conv_s.read())
    Conv2D_valid = Template(conv_v.read())
    Add = Template(ad.read())
    Dense_softmax = Template(den_s.read())
    Dense_relu = Template(den_r.read())
    Flatten = Template(fla.read())
    GlobalAveragePooling = Template(gap.read())
    GlobalMaxPooling = Template(gmp.read())
    MaxPooling2D = Template(mxp.read())
    AveragePooling2D = Template(avp.read())
    Relu = Template(rl.read())
    Softmax = Template(sm.read())
    ZeroPadding = Template(zp.read())
    model = Template(m.read())
    Conv2D_same_hw = Template(hw_conv_s.read()) #HW conv2D_same
    MaxPooling2D_hw = Template(hw_maxp.read()) #HW maxp2D
    stream_io = Template(st.read())
    stream_io_dense = Template(st_d.read())
    top_func = Template(t_func.read())
    top = Template(t.read())
    top_func_dense = Template(t_func_dense.read())
    top_dense = Template(t_dense.read())
    
    sw = Template(sw.read())
   
    # Main 3) Read Layer Information from CSV

    csv_file = open(sys.argv[1])
    csv_reader = csv.DictReader(csv_file)
    model_name=sys.argv[2]

    # Main 4) Generate Function depending on layer_type

    #File Pointers

    for row in csv_reader:
        #Count Line number
        line_count+= 1;
        line_num_str=str(line_count)
        #Get Input, Output shape
        input_shape = row["batch_input_shape"][1 : -1].split(", ")
        output_shape = row["batch_output_shape"][1 : -1].split(", ")
         #Get first_input_shape
        if line_count==0 :
            first_input_shape=row["batch_input_shape"][1 : -1].split(", ")
        # layer_type = convolution2D(I,O,B,W) (padding option: valid , same)
        if row["layer_type"] == 'Conv2D' :
            #Count Conv2D number
            conv_count+=1
            filter_shape = row["kernel_size"][1:-1].split(", ")
            stride_shape = row["strides"][1:-1].split(", ")
            l = {'Name' : row["name"], 'Input_channel' : input_shape[3], 'Input_width' : input_shape[1],'Stride_width': stride_shape[0],'Stride_height':stride_shape[1],
             'Input_height' : input_shape[2], 'Output_channel' : output_shape[3],'Filter_width' : filter_shape[0],
             'Filter_height' : filter_shape[1], 'Output_width' : output_shape[1], 'Output_height' : output_shape[2]}
            lm = {'num': conv_count, 'Name' : row["name"], 'Input_channel' : input_shape[3], 'Output_channel' : output_shape[3],'Filter_width' : filter_shape[0],
             'Filter_height' : filter_shape[1]}
            #SW_static_variables(W,O,B)
            SW_static_v += "static DATA_T W"+line_num_str+"["+ output_shape[3] + "][" + input_shape[3] + "][" + filter_shape[0] + "][" + filter_shape[1] + "];\n"
            SW_static_v += "static DATA_T O"+line_num_str+ "_SW[" + output_shape[3] + "][" + output_shape[1] + "][" + output_shape[2] + "];\n" 
            SW_static_v += "static DATA_T B"+line_num_str + "[" + output_shape[3] + "];\n"
            Output_variables += "static DATA_T O"+line_num_str+ "_SW[" + output_shape[3] + "][" + output_shape[1] + "][" + output_shape[2] + "];\n"
            #SW_variables to sw.txt
            SW_variables += "DATA_T W"+line_num_str+"["+ output_shape[3] + "][" + input_shape[3] + "][" + filter_shape[0] + "][" + filter_shape[1] + "], "
            SW_variables += "DATA_T B"+line_num_str + "[" + output_shape[3] + "],";
            #variables to top_func.txt
            variables += "DATA_T W"+line_num_str+"_i["+ output_shape[3] + "][" + input_shape[3] + "][" + filter_shape[0] + "][" + filter_shape[1] + "], "
            variables += "DATA_T B"+line_num_str + "_i[" + output_shape[3] + "],";
            #HW_static_variables
            HW_static_v += "hls::stream<DATA_T> O"+line_num_str+"_strm;\n"
            #Static_variables to Cpp.txt
            Static_variables += "static DATA_T W"+line_num_str+"_i["+ output_shape[3] + "][" + input_shape[3] + "][" + filter_shape[0] + "][" + filter_shape[1] + "];\n";
            Static_variables += "static DATA_T B"+line_num_str + "_i[" + output_shape[3] + "];\n";
            #top_func_argument to top.txt
            top_func_argument += "W"+line_num_str+"_i, " +"B"+line_num_str+"_i, "
            #assign_value to top.txt
            assign_value += "B"+line_num_str+"_i_m_loop: for (m=0; m<"+output_shape[3]+"; m++) {\n"+"\tB"+line_num_str+"_i[m] = B"+line_num_str+"[m];\n}\n"       
            assign_value += "W"+line_num_str+"_i_m_loop: for (m=0; m<"+output_shape[3]+"; m++) {\n  W"+line_num_str+"_i_k_loop: for (k=0; k<"+input_shape[3]+"; k++) {\n  W"+line_num_str+"_i_i_loop: for (i=0; i<"+filter_shape[0]+"; i++) {\n  W"+line_num_str+"_i_j_loop: for (j=0; j<"+filter_shape[1]+"; j++) {\n  W"+line_num_str+"_i[m][k][i][j] = W"+line_num_str+"[m][k][i][j];\n      }\n    }\n  }\n}\n"
	    #function def (Padding option)
            if row["padding"] == 'valid' :
                SW_def_func += Conv2D_valid.substitute(l) +"\n"
            else :
                SW_def_func += Conv2D_same.substitute(l) +"\n"
                HW_def_func += Conv2D_same_hw.substitute(lm) +"\n"
            #Function use
            SW_functions += "SW_" + row["name"]+ "(O" +str(line_count-1) +"_SW,O"+line_num_str +  "_SW,B" +line_num_str + ",W"+line_num_str +");\n"     
            if line_count<=1 :
                HW_functions += "HW_" + row["name"]+"(I_strm, W"+line_num_str+", B"+line_num_str+", O"+line_num_str+"_strm);\n"
            else :
                HW_functions += "HW_" + row["name"]+"(O"+str(line_count-1)+"_strm, W"+line_num_str+", B"+line_num_str+", O"+line_num_str+"_strm);\n"
            #Optimized code
            Optimized_code += "#pragma HLS ARRAY_PARTITION variable=W"+line_num_str+"_i complete dim=1\n"
            Optimized_code += "#pragma HLS ARRAY_PARTITION variable=W"+line_num_str+"_i complete dim=3\n"
            Optimized_code += "#pragma HLS ARRAY_PARTITION variable=W"+line_num_str+"_i complete dim=4\n" 
            Optimized_code += "#pragma HLS ARRAY_PARTITION variable=B"+line_num_str+"_i complete\n" 
            #Stream declaration
            variable_name="O"+line_num_str+"_strm"                                                                  
            Stream_declaration += "hls::stream<DATA_T> "+variable_name+"(\""+variable_name+"\");\n"		
        # layer_type = BatchNormalization(I, O) 
        elif row["layer_type"] == 'BatchNormalization' :
            l = {'Name' : row["name"], 'Input_channel' : input_shape[3], 'Input_width' : input_shape[1],
             'Input_height' : input_shape[2], 'Output_channel' : output_shape[3],
              'Output_width' : output_shape[1], 'Output_height' : output_shape[2]}
            #SW_static_variables (O)
            SW_static_v += "static DATA_T O"+line_num_str+ "_SW[" + output_shape[3] + "][" + output_shape[1] + "][" + output_shape[2] + "];\n"
            Output_variables += "static DATA_T O"+line_num_str+ "_SW[" + output_shape[3] + "][" + output_shape[1] + "][" + output_shape[2] + "];\n"         
	    #function def
            SW_def_func += BatchNormalization.substitute(l) +"\n"
            #Function use
            SW_functions += "SW_" + row["name"]+ "(O" +str(line_count-1) +"_SW,O"+line_num_str +  "_SW);\n"  
            #Stream declaration
            variable_name="O"+line_num_str+"_strm"                                                                  
            Stream_declaration += "hls::stream<DATA_T> "+variable_name+"(\""+variable_name+"\");\n"
    
        # layer_type = Activation(Relu)(I, O)
        elif row["layer_type"] == 'Activation' :
            l = {'Name' : row["name"], 'Input_channel' : input_shape[3], 'Input_width' : input_shape[1],
             'Input_height' : input_shape[2], 'Output_channel' : output_shape[3],
              'Output_width' : output_shape[1], 'Output_height' : output_shape[2]}
            #SW_static_variables (O)
            SW_static_v += "static DATA_T O"+line_num_str+ "_SW[" + output_shape[3] + "][" + output_shape[1] + "][" + output_shape[2] + "];\n"
            Output_variables += "static DATA_T O"+line_num_str+ "_SW[" + output_shape[3] + "][" + output_shape[1] + "][" + output_shape[2] + "];\n"
            #function def
            SW_def_func += Relu.substitute(l) +"\n"
            #Function use
            SW_functions += "SW_" + row["name"]+ "(O" +str(line_count-1) +"_SW,O"+line_num_str +  "_SW);\n"   
            #Stream declaration
            variable_name="O"+line_num_str+"_strm"                                                                  
            Stream_declaration += "hls::stream<DATA_T> "+variable_name+"(\""+variable_name+"\");\n"

        # layer_type = MaxPooling2D (I, O) 
        elif row["layer_type"] == 'MaxPooling2D' :
            pool_count += 1
            pool_shape = row["pool_size"][1:-1].split(", ")
            stride_shape = row["strides"][1:-1].split(", ")
            l = {'Name' : row["name"], 'Input_channel' : input_shape[3], 'Input_width' : input_shape[1],
             'Input_height' : input_shape[2], 'Output_channel' : output_shape[3],
              'Output_width' : output_shape[1], 'Output_height' : output_shape[2], 'Stride_width' : stride_shape[0],
            'Stride_height':stride_shape[1], 'Pool_width' : pool_shape[0], 'Pool_height' : pool_shape[1]}
            lm = {'num': pool_count, 'Name' : row["name"], 'Input_channel' : input_shape[3], 'Output_channel' : output_shape[3],'Filter_width' : filter_shape[0],
             'Filter_height' : filter_shape[1]}
            #SW_static_variables (O)
            SW_static_v += "static DATA_T O"+line_num_str+ "_SW[" + output_shape[3] + "][" + output_shape[1] + "][" + output_shape[2] + "];\n"; 
            HW_static_v += "hls::stream<DATA_T> O"+line_num_str+"_strm;\n"
            Output_variables += "static DATA_T O"+line_num_str+ "_SW[" + output_shape[3] + "][" + output_shape[1] + "][" + output_shape[2] + "];\n"
            #function def
            SW_def_func += MaxPooling2D.substitute(l) +"\n"
            HW_def_func += MaxPooling2D_hw.substitute(lm) +"\n"
            #Function use
            SW_functions += "SW_" + row["name"]+ "(O" +str(line_count-1) +"_SW,O"+line_num_str +  "_SW);\n"   
            HW_functions += "HW_" + row["name"]+"(O"+str(line_count-1)+"_strm, O"+line_num_str+"_strm);\n"
            #Stream declaration
            variable_name="O"+line_num_str+"_strm"                                                                  
            Stream_declaration += "hls::stream<DATA_T> "+variable_name+"(\""+variable_name+"\");\n"                                                                  
        # layer_type = AveragePooling2D (I, O) 
        elif row["layer_type"] == 'AveragePooling2D' :
            pool_shape = row["pool_size"][1:-1].split(", ")
            stride_shape = row["strides"][1:-1].split(", ")
            l = {'Name' : row["name"], 'Input_channel' : input_shape[3], 'Input_width' : input_shape[1],
                 'Input_height' : input_shape[2], 'Output_channel' : output_shape[3],
              'Output_width' : output_shape[1], 'Output_height' : output_shape[2], 'Stride_width' : stride_shape[0],
            'Stride_height':stride_shape[1], 'Pool_width' : pool_shape[0], 'Pool_height' : pool_shape[1]}
            #SW_static_variables (O)
            SW_static_v += "static DATA_T O"+line_num_str+ "_SW[" + output_shape[3] + "][" + output_shape[1] + "][" + output_shape[2] + "];\n"; 
            Output_variables += "static DATA_T O"+line_num_str+ "_SW[" + output_shape[3] + "][" + output_shape[1] + "][" + output_shape[2] + "];\n"    
            #function def
            SW_def_func += AveragePooling2D.substitute(l) +"\n"
            #Function use
            SW_functions += "SW_" + row["name"]+ "(O" +str(line_count-1) +"_SW,O"+line_num_str +  "_SW);\n"  
            #Stream declaration
            variable_name="O"+line_num_str+"_strm"                                                                  
            Stream_declaration += "hls::stream<DATA_T> "+variable_name+"(\""+variable_name+"\");\n"

        # layer_type = Add(I1,I2,O)
        elif row["layer_type"] == 'Add' :
            l = {'Name' : row['name'], 'Input_channel1' : output_shape[3], 'Input_width1' : output_shape[1],
                 'Input_height1' : output_shape[2],'Input_channel2' : output_shape[3], 'Input_width2' : output_shape[1],
                 'Input_height2' : output_shape[2],'Output_channel' : output_shape[3], 'Output_width' : output_shape[1],
             'Output_height' : output_shape[2]}
            #SW_static_variables (O)
            SW_static_v += "static DATA_T O"+line_num_str+ "_SW[" + output_shape[3] + "][" + output_shape[1] + "][" + output_shape[2] + "];\n"; 
            #function def
            SW_def_func += Add.substitute(l) +"\n"
            #Function use
            SW_functions += "SW_" + row["name"]+ "(O" +str(line_count-1) +"_SW,O"+str(line_count-11) +  "_SW,O" +line_num_str+"_SW);\n"   
            #Stream declaration
            variable_name="O"+line_num_str+"_strm"                                                                  
            Stream_declaration += "hls::stream<DATA_T> "+variable_name+"(\""+variable_name+"\");\n"
 
       # layer_type = ZeroPadding2D(I,O)
        elif row["layer_type"] =="ZeroPadding2D":
            padding_shape = row["padding"][1:-1].split(", (")
            padding = padding_shape[0][1:-1].split(", ")
            l = {'Name' : row["name"], 'Input_channel' : input_shape[3], 'Input_width' : input_shape[1],
             'Input_height' : input_shape[2], 'Output_channel' : output_shape[3],
              'Output_width' : output_shape[1], 'Output_height' : output_shape[2], 'Padding_size': padding[0]}
            #SW_static_variables (O)
            SW_static_v += "static DATA_T O"+line_num_str+ "_SW[" + output_shape[3] + "][" + output_shape[1] + "][" + output_shape[2] + "];\n"; 
            #function def
            SW_def_func += ZeroPadding.substitute(l) +"\n"
            #Function use
            SW_functions += "SW_" + row["name"]+ "(O" +str(line_count-1) +"_SW,O"+line_num_str +  "_SW);\n"   
            #Stream declaration
            variable_name="O"+line_num_str+"_strm"                                                                  
            Stream_declaration += "hls::stream<DATA_T> "+variable_name+"(\""+variable_name+"\");\n"    
        # layer_type = InputLayer
        elif row["layer_type"] =="InputLayer":
            #SW_static_variables (I,O0)
            SW_static_v += "static DATA_T I[" + input_shape[3] + "][" + input_shape[1] + "][" + input_shape[2] + "];\n"
            SW_static_v += "static DATA_T O0_SW[" + input_shape[3] + "][" + input_shape[1] + "][" + input_shape[2] + "];\n"
            #HW_static_variables (I,O0)
            HW_static_v += "hls::stream<DATA_T> I_strm;\n"
            #assign_value to top.txt
            assign_value += "hls::stream<DATA_T> I_strm;\nI_i_k_loop: for (k=0; k<"+input_shape[3]+"; k++) {\n  I_i_x_loop: for (x=0; x<"+input_shape[1]+"; x++) {\n  I_i_y_loop: for (y=0; y<"+input_shape[2]+"; y++) {\n  I_i[k][x][y] = I[k][x][y];\n//I_strm.write(I[k][x][y]);\n    }\n  }\n}\n"
            #Static_variables to Model.txt
            Static_variables += "static DATA_T I_i[" + input_shape[3] + "][" + input_shape[1] + "][" + input_shape[2] + "];\n"
            #variables to top_func.txt
            variables += "DATA_T I["+input_shape[3]+"]["+input_shape[1]+"]["+input_shape[2]+"], "                                           
            #top_func_argument to top.txt                                                                                                                                      
            top_func_argument += "I_i, "
            #SW_variables to sw.txt
            SW_variables += "DATA_T I["+input_shape[3]+"]["+input_shape[1]+"]["+input_shape[2]+"],"                                                                                                                                     
        	#Stream declaration
            Stream_declaration += "hls::stream<DATA_T> I_strm(\"I_strm\");\n"           	
        # layer_type = Flatten(I,O)
        elif row["layer_type"] == "Flatten":
            l = {'Name':row["name"],'Input_channel':input_shape[3],'Input_width':input_shape[1],'Input_height':input_shape[2],'Output_channel':output_shape[1]}
            #SW_static_variables (O)
            SW_static_v += "static DATA_T O"+line_num_str+ "_SW[" + output_shape[1] + "];\n"
            Output_variables += "static DATA_T O"+line_num_str+ "_SW[" + output_shape[1] + "];\n"
            #function def
            SW_def_func += Flatten.substitute(l) + "\n"
            #Function use
            SW_functions += "SW_"+row["name"]+"(O"+str(line_count-1)+"_SW,O"+line_num_str+"_SW);\n"
            #Stream declaration
            variable_name="O"+line_num_str+"_strm"                                                                  
            Stream_declaration += "hls::stream<DATA_T> "+variable_name+"(\""+variable_name+"\");\n"                                                                                                                                      
        # layer_type = Dense(I,W,B,O) (Activation option : relu , softmax)
        elif row["layer_type"] == "Dense":
            l = {'Name':row["name"],'Input_channel':input_shape[1],'Output_channel':output_shape[1]}
            #SW_static_variable (O,W,B)
            SW_static_v += "static DATA_T O"+line_num_str+ "_SW[" + output_shape[1] + "];\n"
            SW_static_v += "static DATA_T W"+line_num_str+ "_SW[" + output_shape[1] + "][" + input_shape[1] + "];\n"
            SW_static_v += "static DATA_T B"+line_num_str+ "_SW[" + output_shape[1] + "];\n"
            Output_variables += "static DATA_T O"+line_num_str+ "_SW[" + output_shape[1] + "];\n"
            #Static_variables to Cpp.txt
            Static_variables += "static DATA_T W"+line_num_str+"_i["+ output_shape[1] + "][" + input_shape[1] + "];\n";
            Static_variables += "static DATA_T B"+line_num_str + "_i[" + output_shape[1] + "];\n";
	    #variables to top_func.txt
            variables += "DATA_T W"+line_num_str+"_i["+ output_shape[1] + "][" + input_shape[1] + "], "
            variables += "DATA_T B"+line_num_str + "_i[" + output_shape[1] + "],";
            #assign_value to top.txt
            assign_value += "B"+line_num_str+"_i_m_loop: for (m=0; m<"+output_shape[1]+"; m++) {\n"+"\tB"+line_num_str+"_i[m] = B"+line_num_str+"[m];\n}\n"       
            assign_value += "W"+line_num_str+"_i_m_loop: for (m=0; m<"+output_shape[1]+"; m++) {\n  W"+line_num_str+"_i_k_loop: for (k=0; k<"+input_shape[1]+"; k++) {\n  W"+line_num_str+"_i[m][k] = W"+line_num_str+"[m][k];\n  }\n}\n"
        	#function def 
            if row["activation"] == 'relu' : # Activation = relu
                SW_def_func += Dense_relu.substitute(l) + "\n"
            else :  # Activation = softmax
                SW_def_func += Dense_softmax.substitute(l) + "\n"
            HW_def_func += Dense_softmax.substitute(l) + "\n"
            #functipn use
            SW_functions += "SW_"+row["name"]+"(O"+str(line_count-1)+"_SW,W"+line_num_str+"_SW,B"+line_num_str+"_SW,O"+line_num_str+"_SW);\n"
            if line_count<=1 :
                HW_functions += "HW_" + row["name"]+"(I_strm, W"+line_num_str+", B"+line_num_str+", O"+line_num_str+"_strm);\n"
            else :
                HW_functions += "HW_" + row["name"]+"(O"+str(line_count-1)+"_strm, W"+line_num_str+", B"+line_num_str+", O"+line_num_str+"_strm);\n"
            Optimized_code += "#pragma HLS ARRAY_PARTITION variable=W"+line_num_str+"_i complete dim=1\n"
            Optimized_code += "#pragma HLS ARRAY_PARTITION variable=W"+line_num_str+"_i complete dim=3\n"
            Optimized_code += "#pragma HLS ARRAY_PARTITION variable=W"+line_num_str+"_i complete dim=4\n" 
            Optimized_code += "#pragma HLS ARRAY_PARTITION variable=B"+line_num_str+"_i complete\n"
	    #top_func_argument to top.txt
            top_func_argument += "W"+line_num_str+"_i, " +"B"+line_num_str+"_i, "                                                                  
            #Stream declaration
            variable_name="O"+line_num_str+"_strm"                                                                  
            Stream_declaration += "hls::stream<DATA_T> "+variable_name+"(\""+variable_name+"\");\n" 
	    #SW_variables to sw.txt
            SW_variables += "DATA_T W"+line_num_str+"["+ output_shape[1] + "][" + input_shape[1] + "], "
            SW_variables += "DATA_T B"+line_num_str + "[" + output_shape[1] + "],";               	

    # Make C file

    top_func_argument += "O_i"
    a=Output_variables.rfind('s', 0,len(Output_variables))
    Output_variables=Output_variables[:a]

    # Output shape 3D
    if len(output_shape)>2:
        SW_variables += "DATA_T O"+line_num_str+"_SW["+output_shape[3]+"]["+output_shape[1]+"]["+output_shape[2]+"]"
        variables += "DATA_T O["+output_shape[3]+"]["+output_shape[1]+"]["+output_shape[2]+"]"
	 #Template Stream_io.txt 
        strm={'Input_channel':first_input_shape[3], 'Input_width': first_input_shape[1], 'Input_height': first_input_shape[2], 'Output_channel': output_shape[3],    
	'Output_width':output_shape[1], 'Output_height':output_shape[2]}
        stream_template=stream_io.substitute(strm)+"\n"	
	#Template top_func.txt
        tf={'model_name':model_name,'Input_channel':first_input_shape[3] , 'Input_width': first_input_shape[1] , 'Input_height': first_input_shape[2], 'Output_channel': output_shape[3], 'Output_width': output_shape[1], 'Output_height': output_shape[2],'variables': variables,
        'Optimized_code':Optimized_code ,'Stream_declaration':Stream_declaration , 'Function_call': HW_functions}
        top_func_template=top_func.substitute(tf)
        #Template top.txt
        to={'model_name':model_name, 'variables':variables, 'Output_channel':output_shape[3],'Output_width':output_shape[1], 
        'Output_height':output_shape [2], 'assign_value':assign_value, 'top_func_argument':top_func_argument}
        top_template=top.substitute(to)
    
    # Output shape 1D
    else:
        SW_variables += "DATA_T O"+line_num_str+"_SW["+output_shape[1]+"]"
        variables += "DATA_T O["+output_shape[1]+"]"
 	#Template Stream_io.txt    
        strm={'Input_channel':first_input_shape[3], 'Input_width': first_input_shape[1], 'Input_height': first_input_shape[2], 'last_output_channel': output_shape[1]}
        stream_template=stream_io_dense.substitute(strm)+"\n"
        #Template top_func.txt
        tf={'model_name':model_name,'Input_channel':first_input_shape[3] , 'Input_width': first_input_shape[1] , 'Input_height': first_input_shape[2],
        'last_output_channel': output_shape[1], 'variables': variables, 'Optimized_code':Optimized_code ,'Stream_declaration':Stream_declaration , 
	'Function_call': HW_functions}       
        top_func_template=top_func_dense.substitute(tf)
        #Template top.txt
        to={'model_name':model_name, 'variables':variables, 'last_output_channel':output_shape[1],'assign_value':assign_value, 
        'top_func_argument':top_func_argument}
        top_template=top_dense.substitute(to)
    
    #Template sw.txt
    s={'SW_variables': SW_variables,'Output_variables': Output_variables,'SW_functions':SW_functions}
    sw_template=sw.substitute(s)
    #Template Cpp.txt
    f = {'D_type':sys.argv[3], 'Stream_io':stream_template, 'Static_variables': Static_variables, 'top_func':top_func_template, 'top':top_template, 'sw':sw_template, 'SW_def_func':SW_def_func, 'HW_def_func':HW_def_func}
    c_file = model.substitute(f) + "\n";

    file = open('Output/'+model_name+'.cpp','w')
    file.write(c_file)
    file.close()


