
# coding: utf-8

# Import

# In[95]:


import csv
import sys
from string import Template
global last_output_row
global first_input_shape


# Main 1) Define variable

# In[96]:


if __name__ == "__main__":
    SW_def_func =""
    HW_def_func =""
    SW_functions = ""
    HW_functions = ""
    SW_static_v = ""
    HW_static_v = ""
    Static_variables="" #Argument to Model.txt
    vgg19_argument="" #Argument to vgg19_top.txt
    Output_variables="" #Argument to vgg19_sw.txt
    SW_variables="" #Argument to vgg19_sw.txt
    assign_value="" #Argument to vgg19_top.txt
    Optimized_code="" #Argument to vgg19.txt
    variables="" #Argument to vgg19.txt
    Stream_declaration="" #Argument to vgg19.txt
    line_count = -1
    conv_count=0
    pool_count = 0
    

# Main 2) Load Template

# In[97]:


# Open file
batch_normal = open("Template/BatchNormalization.txt")
conv_s = open("Template/Conv2D_same.txt")
conv_v = open("Template/Conv2D_valid.txt")
ad = open("Template/Add.txt")
den_s = open("Template/Dense_Softmax.txt")
den_r = open("Template/Dense_Relu.txt")
fla = open("Template/Flatten.txt")
gap = open("Template/GlobalAveragePooling.txt")
gmp = open("Template/GlobalMaxPooling.txt")
mxp = open("Template/MaxPooling2D.txt")
avp = open("Template/AveragePooling2D.txt")
rl = open("Template/Relu.txt")
sm = open("Template/Softmax.txt")
zp = open("Template/ZeroPadding.txt")
m = open("Template/Model.txt")
st=open("Template/Stream_io.txt")
vgg=open("Template/vgg19.txt")
top=open("Template/vgg19_top.txt")
sw=open("Template/vgg19_sw.txt")

#Open HW file
hw_conv_s = open("Template/Conv2D_same_HW.txt")
hw_conv_v = open("Template/Conv2D_valid_HW.txt")
hw_maxp = open("Template/MaxPooling2D_HW.txt")
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
vgg19 = Template(vgg.read())
vgg19_top = Template(top.read())
vgg19_sw = Template(sw.read())


# Main 3) Read Layer Information from CSV

# In[98]:


csv_file = open(sys.argv[1])
csv_reader = csv.DictReader(csv_file)
model_name=sys.argv[2]

# Main 4) Generate Function depending on layer_type

# In[99]:


#File Pointers

for row in csv_reader:
    #Count Line number
    line_count+= 1;
    #Get Input, Output shape
    input_shape = row["batch_input_shape"][1 : -1].split(", ")
    output_shape = row["batch_output_shape"][1 : -1].split(", ")
    #Get last_output_row
    last_output_row=row
    #Get first_input_shape
    if line_count==0:
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
        SW_static_v += "static DATA_T W"+str(line_count)+"["+ output_shape[3] + "][" + input_shape[3] + "][" + filter_shape[0] + "][" + filter_shape[1] + "];\n";
        SW_static_v += "static DATA_T O"+str(line_count)+ "_SW[" + output_shape[3] + "][" + output_shape[1] + "][" + output_shape[2] + "];\n"; 
        SW_static_v += "static DATA_T B"+str(line_count) + "[" + output_shape[3] + "];\n";
        Output_variables += "static DATA_T O"+str(line_count)+ "_SW[" + output_shape[3] + "][" + output_shape[1] + "][" + output_shape[2] + "];\n"
        #SW_variables to vgg19_sw.txt
        SW_variables += "DATA_T W"+str(line_count)+"["+ output_shape[3] + "][" + input_shape[3] + "][" + filter_shape[0] + "][" + filter_shape[1] + "], "
        SW_variables += "DATA_T B"+str(line_count) + "[" + output_shape[3] + "],";
        #variables to vgg19.txt
        variables += "DATA_T W"+str(line_count)+"_i["+ output_shape[3] + "][" + input_shape[3] + "][" + filter_shape[0] + "][" + filter_shape[1] + "], "
        variables += "DATA_T B"+str(line_count) + "_i[" + output_shape[3] + "],";
        #HW_static_variables
        HW_static_v += "hls::stream<DATA_T> O"+str(line_count)+"_strm;\n"
        #Static_variables to Model.txt
        Static_variables += "static DATA_T W"+str(line_count)+"_i["+ output_shape[3] + "][" + input_shape[3] + "][" + filter_shape[0] + "][" + filter_shape[1] + "];\n";
        Static_variables += "static DATA_T B"+str(line_count) + "_i[" + output_shape[3] + "];\n";
        #vgg19_argument to vgg_top.txt
        vgg19_argument += "W"+str(line_count)+"_i, " +"B"+str(line_count)+"_i, "
        #assign_value to vgg_top.txt
        assign_value += "B"+str(line_count)+"_i_m_loop: for (m=0; m<"+output_shape[3]+"; m++) {\n"+"\tB"+str(line_count)+"_i[m] = B"+str(line_count)+"[m];\n}\n"       
        assign_value += "W"+str(line_count)+"_i_m_loop: for (m=0; m<"+output_shape[3]+"; m++) {\n  W"+str(line_count)+"_i_k_loop: for (k=0; k<"+input_shape[3]+"; k++) {\n  W"+str(line_count)+"_i_i_loop: for (i=0; i<"+filter_shape[0]+"; i++) {\n  W"+str(line_count)+"_i_j_loop: for (j=0; j<"+filter_shape[1]+"; j++) {\n  W"+str(line_count)+"_i[m][k][i][j] = W"+str(line_count)+"[m][k][i][j];\n      }\n    }\n  }\n}\n"
	#function def (Padding option)
        if row["padding"] == 'valid' :
            SW_def_func += Conv2D_valid.substitute(l) +"\n"
        else :
            SW_def_func += Conv2D_same.substitute(l) +"\n"
        HW_def_func += Conv2D_same_hw.substitute(lm) +"\n"
        #Function use
        SW_functions += "SW_" + row["name"]+ "(O" +str(line_count-1) +"_SW,O"+str(line_count) +  "_SW,B" +str(line_count) + ",W"+str(line_count) +");\n"     
        if line_count<=1 :
            HW_functions += "HW_" + row["name"]+"(I_strm, W"+str(line_count)+", B"+str(line_count)+", O"+str(line_count)+"_strm);\n"
        else :
            HW_functions += "HW_" + row["name"]+"(O"+str(line_count-1)+"_strm, W"+str(line_count)+", B"+str(line_count)+", O"+str(line_count)+"_strm);\n"
        #Optimized code
        Optimized_code += "#pragma HLS ARRAY_PARTITION variable=W"+str(line_count)+"_i complete dim=2\n"
        Optimized_code += "#pragma HLS ARRAY_PARTITION variable=W"+str(line_count)+"_i complete dim=3\n"
        Optimized_code += "#pragma HLS ARRAY_PARTITION variable=W"+str(line_count)+"_i complete dim=4\n" 
        Optimized_code += "#pragma HLS ARRAY_PARTITION variable=B"+str(line_count)+"_i complete\n" 
        #Stream declaration
        variable_name="O"+str(line_count)+"_strm"                                                                  
        Stream_declaration += "hls::stream<DATA_T> "+variable_name+"(\""+variable_name+"\");\n"
    # layer_type = BatchNormalization(I, O) 
    # if Batch size = None(one Input), Mean = 0, Var = 1 fixed.
    elif row["layer_type"] == 'BatchNormalization' :
        l = {'Name' : row["name"], 'Input_channel' : input_shape[3], 'Input_width' : input_shape[1],
             'Input_height' : input_shape[2], 'Output_channel' : output_shape[3],
              'Output_width' : output_shape[1], 'Output_height' : output_shape[2]}
        #SW_static_variables (O)
        SW_static_v += "static DATA_T O"+str(line_count)+ "_SW[" + output_shape[3] + "][" + output_shape[1] + "][" + output_shape[2] + "];\n"; 
        #function def
        SW_def_func += BatchNormalization.substitute(l) +"\n"
        #Function use
        SW_functions += "SW_" + row["name"]+ "(O" +str(line_count-1) +"_SW,O"+str(line_count) +  "_SW);\n"  
        #Stream declaration
        variable_name="O"+str(line_count)+"_strm"                                                                  
        Stream_declaration += "hls::stream<DATA_T> "+variable_name+"(\""+variable_name+"\");\n"
    # layer_type = Activation(Relu)(I, O)
    elif row["layer_type"] == 'Activation' :
        l = {'Name' : row["name"], 'Input_channel' : input_shape[3], 'Input_width' : input_shape[1],
             'Input_height' : input_shape[2], 'Output_channel' : output_shape[3],
              'Output_width' : output_shape[1], 'Output_height' : output_shape[2]}
        #SW_static_variables (O)
        SW_static_v += "static DATA_T O"+str(line_count)+ "_SW[" + output_shape[3] + "][" + output_shape[1] + "][" + output_shape[2] + "];\n"; 
        #function def
        SW_def_func += Relu.substitute(l) +"\n"
        #Function use
        SW_functions += "SW_" + row["name"]+ "(O" +str(line_count-1) +"_SW,O"+str(line_count) +  "_SW);\n"   
        #Stream declaration
        variable_name="O"+str(line_count)+"_strm"                                                                  
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
        SW_static_v += "static DATA_T O"+str(line_count)+ "_SW[" + output_shape[3] + "][" + output_shape[1] + "][" + output_shape[2] + "];\n"; 
        HW_static_v += "hls::stream<DATA_T> O"+str(line_count)+"_strm;\n"
        Output_variables += "static DATA_T O"+str(line_count)+ "_SW[" + output_shape[3] + "][" + output_shape[1] + "][" + output_shape[2] + "];\n"
        #function def
        SW_def_func += MaxPooling2D.substitute(l) +"\n"
        HW_def_func += MaxPooling2D_hw.substitute(lm) +"\n"
        #Function use
        SW_functions += "SW_" + row["name"]+ "(O" +str(line_count-1) +"_SW,O"+str(line_count) +  "_SW);\n"   
        HW_functions += "HW_" + row["name"]+"(O"+str(line_count-1)+"_strm, O"+str(line_count)+"_strm);\n"
        #Stream declaration
        variable_name="O"+str(line_count)+"_strm"                                                                  
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
        SW_static_v += "static DATA_T O"+str(line_count)+ "_SW[" + output_shape[3] + "][" + output_shape[1] + "][" + output_shape[2] + "];\n"; 
        #function def
        SW_def_func += AveragePooling2D.substitute(l) +"\n"
        #Function use
        SW_functions += "SW_" + row["name"]+ "(O" +str(line_count-1) +"_SW,O"+str(line_count) +  "_SW);\n"  
        #Stream declaration
        variable_name="O"+str(line_count)+"_strm"                                                                  
        Stream_declaration += "hls::stream<DATA_T> "+variable_name+"(\""+variable_name+"\");\n"
    # layer_type = Add(I1,I2,O)
    elif row["layer_type"] == 'Add' :
        l = {'Name' : row['name'], 'Input_channel1' : output_shape[3], 'Input_width1' : output_shape[1],
             'Input_height1' : output_shape[2],'Input_channel2' : output_shape[3], 'Input_width2' : output_shape[1],
             'Input_height2' : output_shape[2],'Output_channel' : output_shape[3], 'Output_width' : output_shape[1],
             'Output_height' : output_shape[2]}
        #SW_static_variables (O)
        SW_static_v += "static DATA_T O"+str(line_count)+ "_SW[" + output_shape[3] + "][" + output_shape[1] + "][" + output_shape[2] + "];\n"; 
        #function def
        SW_def_func += Add.substitute(l) +"\n"
        #Function use
        SW_functions += "SW_" + row["name"]+ "(O" +str(line_count-1) +"_SW,O"+str(line_count-11) +  "_SW,O" +str(line_count)+"_SW);\n"   
        #Stream declaration
        variable_name="O"+str(line_count)+"_strm"                                                                  
        Stream_declaration += "hls::stream<DATA_T> "+variable_name+"(\""+variable_name+"\");\n"
    # layer_type = ZeroPadding2D(I,O)
    elif row["layer_type"] =="ZeroPadding2D":
        padding_shape = row["padding"][1:-1].split(", (")
        padding = padding_shape[0][1:-1].split(", ")
        l = {'Name' : row["name"], 'Input_channel' : input_shape[3], 'Input_width' : input_shape[1],
             'Input_height' : input_shape[2], 'Output_channel' : output_shape[3],
              'Output_width' : output_shape[1], 'Output_height' : output_shape[2], 'Padding_size': padding[0]}
        #SW_static_variables (O)
        SW_static_v += "static DATA_T O"+str(line_count)+ "_SW[" + output_shape[3] + "][" + output_shape[1] + "][" + output_shape[2] + "];\n"; 
        #function def
        SW_def_func += ZeroPadding.substitute(l) +"\n"
        #Function use
        SW_functions += "SW_" + row["name"]+ "(O" +str(line_count-1) +"_SW,O"+str(line_count) +  "_SW);\n"   
        #Stream declaration
        variable_name="O"+str(line_count)+"_strm"                                                                  
        Stream_declaration += "hls::stream<DATA_T> "+variable_name+"(\""+variable_name+"\");\n"    
    # layer_type = InputLayer
    elif row["layer_type"] =="InputLayer":
        #SW_static_variables (I,O0)
        SW_static_v += "static DATA_T I[" + input_shape[3] + "][" + input_shape[1] + "][" + input_shape[2] + "];\n"
        SW_static_v += "static DATA_T O0_SW[" + input_shape[3] + "][" + input_shape[1] + "][" + input_shape[2] + "];\n"
        #HW_static_variables (I,O0)
        HW_static_v += "hls::stream<DATA_T> I_strm;\n"
        #assign_value to vgg_top.txt
        assign_value += "hls::stream<DATA_T> I_strm;\nI_i_k_loop: for (k=0; k<"+input_shape[3]+"; k++) {\n  I_i_x_loop: for (x=0; x<"+input_shape[1]+"; x++) {\n  I_i_y_loop: for (y=0; y<"+input_shape[2]+"; y++) {\n  I_i[k][x][y] = I[k][x][y];\n//I_strm.write(I[k][x][y]);\n    }\n  }\n}\n"
        #Static_variables to Model.txt
        Static_variables += "static DATA_T I_i[" + input_shape[3] + "][" + input_shape[1] + "][" + input_shape[2] + "];\n"
        #variables to vgg19.txt
        variables += "DATA_T I["+input_shape[3]+"]["+input_shape[1]+"]["+input_shape[2]+"], "                                                                                                                                     
        #vgg19_argument to vgg19_top.txt                                                                                                                                      
        vgg19_argument += "I_i, "
        #SW_variables to vgg19_sw.txt
        SW_variables += "DATA_T I["+input_shape[3]+"]["+input_shape[1]+"]["+input_shape[2]+"], "                                                                                                                                     
	#Stream declaration
        Stream_declaration += "hls::stream<DATA_T> I_strm(\"I_strm\");\n"                                                                                                                                      
    # layer_type = Flatten(I,O)
    elif row["layer_type"] == "Flatten":
        l = {'Name':row["name"],'Input_channel':input_shape[3],'Input_width':input_shape[1],'Input_height':input_shape[2],'Output_channel':output_shape[1]}
        #SW_static_variables (O)
        SW_static_v += "static DATA_T O"+str(line_count)+ "_SW[" + output_shape[1] + "];\n";
        #function def
        SW_def_func += Flatten.substitute(l) + "\n"
        #Function use
        SW_functions += "SW_"+row["name"]+"(O"+str(line_count-1)+"_SW,O"+str(line_count)+"_SW);\n"
        #Stream declaration
        variable_name="O"+str(line_count)+"_strm"                                                                  
        Stream_declaration += "hls::stream<DATA_T> "+variable_name+"(\""+variable_name+"\");\n"                                                                                                                                      
    # layer_type = Dense(I,W,B,O) (Activation option : relu , softmax)
    elif row["layer_type"] == "Dense":
        l = {'Name':row["name"],'Input_channel':input_shape[1],'Output_channel':output_shape[1]}
        #SW_static_variable (O,W,B)
        SW_static_v += "static DATA_T O"+str(line_count)+ "_SW[" + output_shape[1] + "];\n"
        SW_static_v += "static DATA_T W"+str(line_count)+ "_SW[" + output_shape[1] + "][" + input_shape[1] + "];\n"
        SW_static_v += "static DATA_T B"+str(line_count)+ "_SW[" + output_shape[1] + "];\n"
        #assign_value to vgg_top.txt
        assign_value += "B"+str(line_count)+"_i_m_loop: for (m=0; m<"+output_shape[1]+"; m++) {\n"+"\tB"+line_count+"_i[m] = B"+str(line_count)+"[m];\n}\n"       
        assign_value += "W"+str(line_count)+"_i_m_loop: for (m=0; m<"+output_shape[1]+"; m++) {\n  W"+str(line_count)+"_i_k_loop: for (k=0; k<"+input_shape[1]+"; k++) {\n  W"+str(line_count)+"_i[m][k] = W"+str(line_count)+"[m][k];\n  }\n}\n"
	#function def 
        if row["activation"] == 'relu' : # Activation = relu
            SW_def_func += Dense_relu.substitute(l) + "\n"
        else :  # Activation = softmax
            SW_def_func += Dense_softmax.substitute(l) + "\n"
        #functipn use
        SW_functions += "SW_"+row["name"]+"(O"+str(line_count-1)+"_SW,W"+str(line_count)+"_SW,B"+str(line_count)+"_SW,O"+str(line_count)+"_SW);\n"
        # If this is end of layer, Generate final output O
        if row["activation"] == 'softmax' :
            SW_static_v += "static DATA_T O_SW[" + output_shape[1] + "];\n";
            SW_functions += "O_SW = O"+str(line_count)+"_SW;\n"
        Optimized_code += "#pragma HLS ARRAY_PARTITION variable=W"+str(line_count)+"_i complete dim=2\n"
        Optimized_code += "#pragma HLS ARRAY_PARTITION variable=W"+str(line_count)+"_i complete dim=3\n"
        Optimized_code += "#pragma HLS ARRAY_PARTITION variable=W"+str(line_count)+"_i complete dim=4\n" 
        Optimized_code += "#pragma HLS ARRAY_PARTITION variable=B"+str(line_count)+"_i complete\n"                                                                  
        #Stream declaration
        variable_name="O"+str(line_count)+"_strm"                                                                  
        Stream_declaration += "hls::stream<DATA_T> "+variable_name+"(\""+variable_name+"\");\n"    
    else :
        print ('Not defined')
    
    


# Make C file

# In[100]:


vgg19_argument += "O_i"
a=Output_variables.rfind('D', 0,len(Output_variables))
Output_variables=Output_variables[:a]
last_output_shape = last_output_row["batch_output_shape"][1 : -1].split(", ")
SW_variables += "DATA_T O"+str(line_count)+"_SW["+last_output_shape[3]+"]["+last_output_shape[1]+"]["+last_output_shape[2]+"]"
variables += "DATA_T O["+last_output_shape[3]+"]["+last_output_shape[1]+"]["+last_output_shape[2]+"]"
#Stream_io Template
strm={'Input_channel':first_input_shape[3], 'Input_width': first_input_shape[1], 'Input_height': first_input_shape[2], 'Output_channel': last_output_shape[3], 'Output_width':last_output_shape[1], 'Output_height':last_output_shape[2]}
stream_template=stream_io.substitute(strm)+"\n"
#vgg19
vg={'Input_channel':first_input_shape[3] , 'Input_width': first_input_shape[1] , 'Input_height': first_input_shape[2], 'Output_channel': last_output_shape[3], 'Output_width': last_output_shape[1], 'Output_height': last_output_shape[2],'variables': variables, 'Optimized_code':Optimized_code , 'Stream_declaration':Stream_declaration , 'Function_call': HW_functions}
vgg19_template=vgg19.substitute(vg)
#vgg19_top
top={'variables':SW_static_v, 'Output_channel':last_output_shape[3],'Output_width':last_output_shape[1], 'Output_height':last_output_shape[2], 'assign_value':assign_value, 'vgg19_argument':vgg19_argument}
vgg19top_template=vgg19_top.substitute(top)
#vgg19_sw
vg_s={'SW_variables': SW_variables,'Output_variables': Output_variables,'SW_functions':SW_functions}
vgg19sw_template=vgg19_sw.substitute(vg_s)
#model
f = {'Stream_io':stream_template, 'Static_variables': Static_variables, 'VGG19':vgg19_template, 'VGG19_top':vgg19top_template, 'VGG19_sw':vgg19sw_template, 'SW_def_func':SW_def_func, 'HW_def_func':HW_def_func}
c_file = model.substitute(f) + "\n";
print (c_file)

file = open(model_name+'.cpp','w')
file.write(c_file)
file.close()

