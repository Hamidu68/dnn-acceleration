
# coding: utf-8

# Import
import csv
import sys
from string import Template

#sys.argv[1]=Test.csv
#sys.argv[2]=model name
#sys.argv[3]=Data type
#sys.argv[4]=Relu(True and False)

if __name__ == "__main__":
    # Main 1) Define variable

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
    variables="" #Argument to top.txt
    variables_i = "" #Argument to top_func.txt
    Stream_declaration="" #Argument to Cpp.txt
    line_count = -1
    conv_count=0
    pool_count = 0
    global input_shape
    global output_shape
    global first_input_shape

    # Main 2) Load Template

    # Open file
    #with relu
    if sys.argv[4] == "True":
        conv_s = open("../Template/Function/Conv2D_same_relu.txt")
    #without relu
    else :
        conv_s = open("../Template/Function/Conv2D_same.txt")
    mxp = open("../Template/Function/MaxPooling2D.txt")
    m = open("../Template/Main/Cpp.txt")
    st=open("../Template/Function/Stream_io.txt")
    t_func=open("../Template/Function/top_func.txt")
    t=open("../Template/Function/top.txt")
    sw=open("../Template/Function/sw.txt")
    hw_conv_s = open("../Template/Function/Conv2D_same_DAC2017.txt")
    hw_maxp = open("../Template/Function/MaxPooling2D_HW.txt")
    # Read Template
    Conv2D_same = Template(conv_s.read())
    MaxPooling2D = Template(mxp.read())
    model = Template(m.read())
    Conv2D_same_hw = Template(hw_conv_s.read()) #HW conv2D_same
    MaxPooling2D_hw = Template(hw_maxp.read()) #HW maxp2D
    stream_io = Template(st.read())
    top_func = Template(t_func.read())
    top = Template(t.read())
    sw = Template(sw.read())

    # Main 3) Read Layer Information from CSV

    csv_file = open(sys.argv[1])
    csv_reader = csv.DictReader(csv_file)
    model_name=sys.argv[2]

    # Main 4) Generate Function depending on layer_type

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
            lm = {'num': conv_count, 'Name' : "HW_"+row["name"], 'Input_channel' : input_shape[3], 'Output_channel' : output_shape[3], 'Input_width' : input_shape[1], 'Output_width' : output_shape[1], 'Output_height' : output_shape[2], 'Input_height' : input_shape[2],'Filter_width' : filter_shape[0], 'Filter_height' : filter_shape[1]}
            #SW_static_variables(W,O,B)
            SW_static_v += "static DATA_T W"+line_num_str+"["+ output_shape[3] + "][" + input_shape[3] + "][" + filter_shape[0] + "][" + filter_shape[1] + "];\n"
            SW_static_v += "static DATA_T O"+line_num_str+ "_SW[" + output_shape[3] + "][" + output_shape[1] + "][" + output_shape[2] + "];\n"
            SW_static_v += "static DATA_T B"+line_num_str + "[" + output_shape[3] + "];\n"
            Output_variables += "static DATA_T O"+line_num_str+ "_SW[" + output_shape[3] + "][" + output_shape[1] + "][" + output_shape[2] + "];\n"
            #SW_variables to sw.txt
            SW_variables += "DATA_T W"+line_num_str+"["+ output_shape[3] + "][" + input_shape[3] + "][" + filter_shape[0] + "][" + filter_shape[1] + "], "
            SW_variables += "DATA_T B"+line_num_str + "[" + output_shape[3] + "],";
            #variables to top.txt
            variables += "DATA_T W"+line_num_str+"["+ output_shape[3] + "][" + input_shape[3] + "][" + filter_shape[0] + "][" + filter_shape[1] + "], "
            variables += "DATA_T B"+line_num_str + "[" + output_shape[3] + "],";
            #variables to top_func.txt
            variables_i += "DATA_T W"+line_num_str+"_d["+ output_shape[3] + "][" + input_shape[3] + "][" + filter_shape[0] + "][" + filter_shape[1] + "], "
            variables_i += "DATA_T B"+line_num_str + "_d[" + output_shape[3] + "],";
            #HW_static_variables
            HW_static_v += "hls::stream<DATA_T> O"+line_num_str+"_strm;\n"
            #Static_variables to Cpp.txt
            Static_variables += "static DATA_T W"+line_num_str+"_d["+ output_shape[3] + "][" + input_shape[3] + "][" + filter_shape[0] + "][" + filter_shape[1] + "];\n";
            Static_variables += "static DATA_T B"+line_num_str + "_d[" + output_shape[3] + "];\n";
            #top_func_argument to top.txt
            top_func_argument += "W"+line_num_str+"_d, " +"B"+line_num_str+"_d, "
            #assign_value to top.txt
            assign_value += "B"+line_num_str+"_d_m_loop: for (m=0; m<"+output_shape[3]+"; m++) {\n"+"\tB"+line_num_str+"_d[m] = B"+line_num_str+"[m];\n}\n"
            assign_value += "W"+line_num_str+"_d_m_loop: for (m=0; m<"+output_shape[3]+"; m++) {\n  W"+line_num_str+"_d_k_loop: for (k=0; k<"+input_shape[3]+"; k++) {\n  W"+line_num_str+"_d_i_loop: for (i=0; i<"+filter_shape[0]+"; i++) {\n  W"+line_num_str+"_d_j_loop: for (j=0; j<"+filter_shape[1]+"; j++) {\n  W"+line_num_str+"_d[m][k][i][j] = W"+line_num_str+"[m][k][i][j];\n      }\n    }\n  }\n}\n"
            SW_def_func += Conv2D_same.substitute(l) +"\n"
            HW_def_func += Conv2D_same_hw.substitute(lm) +"\n"
            #Function use
            if line_count<=1 :
                HW_functions += "DAC2017_HW_" + row["name"]+"(I_strm, W"+line_num_str+"_d, B"+line_num_str+"_d, O"+line_num_str+"_strm);\n"
                SW_functions += "SW_" + row["name"]+ "(I,O"+line_num_str +  "_SW,B" +line_num_str + ",W"+line_num_str +");\n"
            else :
                SW_functions += "SW_" + row["name"]+ "(O" +str(line_count-1) +"_SW,O"+line_num_str +  "_SW,B" +line_num_str + ",W"+line_num_str +");\n"
                HW_functions += "DAC2017_HW_" + row["name"]+"(O"+str(line_count-1)+"_strm, W"+line_num_str+"_d, B"+line_num_str+"_d, O"+line_num_str+"_strm);\n"
            #Optimized code
            Optimized_code += "#pragma HLS ARRAY_PARTITION variable=W"+line_num_str+"_d complete dim=1\n"
            Optimized_code += "#pragma HLS ARRAY_PARTITION variable=W"+line_num_str+"_d complete dim=3\n"
            Optimized_code += "#pragma HLS ARRAY_PARTITION variable=W"+line_num_str+"_d complete dim=4\n"
            Optimized_code += "#pragma HLS ARRAY_PARTITION variable=B"+line_num_str+"_d complete\n"
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
            lm = {'num': pool_count, 'Name' : row["name"], 'Input_channel' : input_shape[3], 'Input_width' : input_shape[1],
             'Input_height' : input_shape[2] }
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

        # layer_type = InputLayer
        elif row["layer_type"] =="InputLayer":
            #SW_static_variables
            SW_static_v += "static DATA_T I[" + input_shape[3] + "][" + input_shape[1] + "][" + input_shape[2] + "];\n"
            SW_static_v += "static DATA_T O0_SW[" + input_shape[3] + "][" + input_shape[1] + "][" + input_shape[2] + "];\n"
            #HW_static_variables
            HW_static_v += "hls::stream<DATA_T> I_strm;\n"
            #assign_value to top.txt
            assign_value += "hls::stream<DATA_T> I_strm;\nI_d_k_loop: for (k=0; k<"+input_shape[3]+"; k++) {\n  I_d_x_loop: for (x=0; x<"+input_shape[1]+"; x++) {\n  I_d_y_loop: for (y=0; y<"+input_shape[2]+"; y++) {\n  I_d[k][x][y] = I[k][x][y];\n//I_strm.write(I[k][x][y]);\n    }\n  }\n}\n"
            #Static_variables to Model.txt
            Static_variables += "static DATA_T I_d[" + input_shape[3] + "][" + input_shape[1] + "][" + input_shape[2] + "];\n"
            #variables to top.txt
            variables += "DATA_T I["+input_shape[3]+"]["+input_shape[1]+"]["+input_shape[2]+"], "
            #variables to top_func.txt
            variables_i += "DATA_T I["+input_shape[3]+"]["+input_shape[1]+"]["+input_shape[2]+"], "
            #top_func_argument to top.txt
            top_func_argument += "I_d, "
            #SW_variables to sw.txt
            SW_variables += "DATA_T I["+input_shape[3]+"]["+input_shape[1]+"]["+input_shape[2]+"],"
        	#Stream declaration
            Stream_declaration += "hls::stream<DATA_T> I_strm(\"I_strm\");\n"

     # Make C file

    top_func_argument += "O_i"
    SW_variables += "DATA_T O["+output_shape[3]+"]["+output_shape[1]+"]["+output_shape[2]+"]"
    variables += "DATA_T O["+output_shape[3]+"]["+output_shape[1]+"]["+output_shape[2]+"]"
    variables_i += "DATA_T O["+output_shape[3]+"]["+output_shape[1]+"]["+output_shape[2]+"]"
	#Template Stream_io.txt
    strm={'Input_channel':first_input_shape[3], 'Input_width': first_input_shape[1], 'Input_height': first_input_shape[2], 'Output_channel':output_shape[3],'Output_width':output_shape[1], 'Output_height':output_shape[2]}
    stream_template=stream_io.substitute(strm)+"\n"
	#Template top_func.txt
    tf={'model_name':"DAC2017_"+model_name,'variables': variables_i,'Optimized_code':Optimized_code ,'Stream_declaration':Stream_declaration , 'Function_call': HW_functions, 'line_num':line_num_str}
    top_func_template=top_func.substitute(tf)
    #Template top.txt
    to={'model_name':"DAC2017_"+model_name, 'variables':variables, 'Output_channel':output_shape[3],'Output_width':output_shape[1],'Output_height':output_shape [2], 'assign_value':assign_value, 'top_func_argument':top_func_argument}
    top_template=top.substitute(to)

    #Template sw.txt
    s={'model_name':model_name,'SW_variables': SW_variables,'Output_variables': Output_variables,'SW_functions':SW_functions}
    sw_template=sw.substitute(s)
    #Template Cpp.txt
    f = {'D_type':sys.argv[3], 'Stream_io':stream_template, 'Static_variables': Static_variables, 'top_func':top_func_template, 'top':top_template, 'sw':sw_template, 'SW_def_func':SW_def_func, 'HW_def_func':HW_def_func}
    c_file = model.substitute(f) + "\n";

    file = open('Output/DAC2017.cpp','w')
    file.write(c_file)
    file.close()


