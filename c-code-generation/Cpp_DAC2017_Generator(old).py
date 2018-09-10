
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
    DAC_def_func =""
    DAC_functions = ""
    DAC_static_v = ""
    HW_static_v = ""
    Static_variables="" #Argument to Cpp.txt
    DAC_func_argument="" #Argument to DAC.txt
    Output_variables="" #Argument to sw.txt
    DAC_assign_value = "" #Argument to DAC.txt
    DAC_Optimized_code =""
    variables="" #Argument to top.txt
    variables_dac = "" #Argument to DAC.txt
    Stream_declaration="" #Argument to Cpp.txt
    line_count = -1
    conv_count=0
    pool_count = 0
    global input_shape
    global output_shape
    global first_input_shape

    # Main 2) Load Template

    # Open file
    m = open("../Template/Main/Cpp_DAC2017.txt")
    st=open("../Template/Function/Stream_io.txt")
    st_d=open("../Template/Function/Stream_io_dense.txt")
    t_func=open("../Template/Function/top_func.txt")
    t=open("../Template/Function/top.txt")
    t_dense=open("../Template/Function/top_dense.txt")
    hw_conv_our = open("../Template/Function/Conv2D_same_HW.txt")
    hw_conv_dac = open("../Template/Function/Conv2D_same_DAC2017.txt")
    hw_maxp = open("../Template/Function/MaxPooling2D_HW.txt")
    # Read Template
    model = Template(m.read())
    Conv2D_same_hw = Template(hw_conv_our.read()) #HW conv2D_same
    Conv2D_same_DAC2017 = Template(hw_conv_dac.read()) #HW Conv2D_same_DAC2017
    MaxPooling2D_hw = Template(hw_maxp.read()) #HW maxp2D
    stream_io = Template(st.read())
    stream_io_dense = Template(st_d.read())
    top_func = Template(t_func.read())
    top = Template(t.read())
    top_dense = Template(t_dense.read())

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
            lm = {'num': conv_count, 'Name' : row["name"], 'Input_channel' : input_shape[3], 'Output_channel' : output_shape[3], 'Input_width' : input_shape[1],
             'Input_height' : input_shape[2],'Filter_width' : filter_shape[0],'Output_width' : output_shape[1], 'Filter_height' : filter_shape[1]}
            #variables
            variables += "DATA_T W"+line_num_str+"["+ output_shape[3] + "][" + input_shape[3] + "][" + filter_shape[0] + "][" + filter_shape[1] + "], "
            variables += "DATA_T B"+line_num_str + "[" + output_shape[3] + "],";
            #variables for DAC2017 code
            variables_dac += "DATA_T W"+line_num_str+"_DAC["+ output_shape[3] + "][" + input_shape[3] + "][" + filter_shape[0] + "][" + filter_shape[1] + "], "
            variables_dac += "DATA_T B"+line_num_str + "_DAC[" + output_shape[3] + "],";
            #HW_static_variables
            HW_static_v += "hls::stream<DATA_T> O"+line_num_str+"_strm;\n"
            #Static_variables
            Static_variables += "static DATA_T W"+line_num_str+"_DAC["+ output_shape[3] + "][" + input_shape[3] + "][" + filter_shape[0] + "][" + filter_shape[1] + "];\n";
            Static_variables += "static DATA_T B"+line_num_str + "_DAC[" + output_shape[3] + "];\n";
            #function argument
            DAC_func_argument += "W"+line_num_str+"_DAC, " +"B"+line_num_str+"_DAC, "
            #assign_value
            DAC_assign_value += "B"+line_num_str+"_DAC_m_loop: for (m=0; m<"+output_shape[3]+"; m++) {\n"+"\tB"+line_num_str+"_DAC[m] = B"+line_num_str+"[m];\n}\n"
            DAC_assign_value += "W"+line_num_str+"_DAC_m_loop: for (m=0; m<"+output_shape[3]+"; m++) {\n  W"+line_num_str+"_i_k_loop: for (k=0; k<"+input_shape[3]+"; k++) {\n  W"+line_num_str+"_DAC_i_loop: for (i=0; i<"+filter_shape[0]+"; i++) {\n  W"+line_num_str+"_DAC_j_loop: for (j=0; j<"+filter_shape[1]+"; j++) {\n  W"+line_num_str+"_DAC[m][k][i][j] = W"+line_num_str+"[m][k][i][j];\n      }\n    }\n  }\n}\n"
            #Function definition
            DAC_def_func += Conv2D_same_DAC2017.substitute(lm) +"\n"
            #Function use
            if line_count<=1 :
                DAC_functions += "DAC2017_" + row["name"]+"(I_strm, W"+line_num_str+"_DAC, B"+line_num_str+"_DAC, O"+line_num_str+"_strm);\n"
            else :
                DAC_functions += "DAC2017_" + row["name"]+"(O"+str(line_count-1)+"_strm, W"+line_num_str+"_DAC, B"+line_num_str+"_DAC, O"+line_num_str+"_strm);\n"
            #Optimized code
            DAC_Optimized_code += "#pragma HLS ARRAY_PARTITION variable=W"+line_num_str+"_DAC complete dim=1\n"
            DAC_Optimized_code += "#pragma HLS ARRAY_PARTITION variable=W"+line_num_str+"_DAC complete dim=3\n"
            DAC_Optimized_code += "#pragma HLS ARRAY_PARTITION variable=W"+line_num_str+"_DAC complete dim=4\n"
            DAC_Optimized_code += "#pragma HLS ARRAY_PARTITION variable=B"+line_num_str+"_DAC complete\n"
            #Stream declaration
            variable_name="O"+line_num_str+"_strm"
            Stream_declaration += "hls::stream<DATA_T> "+variable_name+"(\""+variable_name+"\");\n"

        # layer_type = MaxPooling2D (I, O)
        elif row["layer_type"] == 'MaxPooling2D' :
            pool_count += 1
            pool_shape = row["pool_size"][1:-1].split(", ")
            stride_shape = row["strides"][1:-1].split(", ")
            lm = {'num': pool_count, 'Name' : row["name"], 'Input_channel' : input_shape[3], 'Input_width' : input_shape[1],
             'Input_height' : input_shape[2] }
            #static_variables
            HW_static_v += "hls::stream<DATA_T> O"+line_num_str+"_strm;\n"
            #function def
            i_def_func += MaxPooling2D_hw.substitute(lm) +"\n"
            #Function use
            DAC_functions += "HW_" + row["name"]+"(O"+str(line_count-1)+"_strm, O"+line_num_str+"_strm);\n"
            #Stream declaration
            variable_name="O"+line_num_str+"_strm"
            Stream_declaration += "hls::stream<DATA_T> "+variable_name+"(\""+variable_name+"\");\n"

        # layer_type = InputLayer
        elif row["layer_type"] =="InputLayer":
            #static_variables
            HW_static_v += "hls::stream<DATA_T> I_strm;\n"
            #assign_value
            DAC_assign_value += "hls::stream<DATA_T> I_strm;\nI_DAC_k_loop: for (k=0; k<"+input_shape[3]+"; k++) {\n  I_DAC_x_loop: for (x=0; x<"+input_shape[1]+"; x++) {\n  I_DAC_y_loop: for (y=0; y<"+input_shape[2]+"; y++) {\n  I_DAC[k][x][y] = I[k][x][y];\n//I_strm.write(I[k][x][y]);\n    }\n  }\n}\n"
            #Static_variables to Model.txt
            Static_variables += "static DATA_T I_DAC[" + input_shape[3] + "][" + input_shape[1] + "][" + input_shape[2] + "];\n"
            #variables
            variables += "DATA_T I["+input_shape[3]+"]["+input_shape[1]+"]["+input_shape[2]+"], "
            #variables to top_func.txt
            variables_dac += "DATA_T I["+input_shape[3]+"]["+input_shape[1]+"]["+input_shape[2]+"], "
            #function argument
            DAC_func_argument += "I_DAC, "
            #Stream declaration
            Stream_declaration += "hls::stream<DATA_T> I_strm(\"I_strm\");\n"
    # Make C file
    DAC_func_argument += "O_i"
    variables += "DATA_T O["+output_shape[3]+"]["+output_shape[1]+"]["+output_shape[2]+"]"
    variables_dac += "DATA_T O["+output_shape[3]+"]["+output_shape[1]+"]["+output_shape[2]+"]"
     #Template Stream_io.txt
    strm={'Input_channel':first_input_shape[3], 'Input_width': first_input_shape[1], 'Input_height': first_input_shape[2], 'Output_channel': output_shape[3],
    'Output_width':output_shape[1], 'Output_height':output_shape[2]}
    stream_template=stream_io.substitute(strm)+"\n"
    #Template for function
    tf={'model_name':"DAC2017",'variables': variables_dac,
        'Optimized_code':DAC_Optimized_code ,'Stream_declaration':Stream_declaration , 'Function_call': DAC_functions, 'line_num':line_num_str}
    DAC_func_template=top_func.substitute(tf)
    #Template sw.txt
    s={'model_name':"DAC2017",'SW_variables': SW_variables,'Output_variables': Output_variables,'SW_functions':SW_functions}
    DAC_sw_template=sw.substitute(s)
   #Template for model function.txt
    to={'model_name':"DAC2017", 'variables':variables, 'Output_channel':output_shape[3],'Output_width':output_shape[1],
        'Output_height':output_shape [2], 'assign_value':DAC_assign_value, 'top_func_argument':DAC_func_argument}
    DAC_top_template=top.substitute(to)
    #Template Cpp_DAC2017.txt
    f = {'D_type':sys.argv[3], 'Stream_io':stream_template, 'Static_variables': Static_variables, 'DAC2017_top' : DAC_top_template, 'DAC2017_func': DAC_func_template, 'DAC2017_def_func': DAC_def_func, 'DAC2017_sw_def_func':, 'DAC2017_sw':DAC_sw_template}
    c_file = model.substitute(f) + "\n";

    file = open('Output/DAC2017_'+model_name+'.cpp','w')
    file.write(c_file)
    file.close()

