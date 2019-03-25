
# coding: utf-8

# In[2]:


import csv
from string import Template
from sys import argv

#sys.argv[1]= Test.csv
#sys.argv[2]=Relu(True and False)

if __name__ == "__main__":
    # Main 1) Define variable

    SW_def_func =""
    SW_functions = ""
    Shared_static_v =""
    SW_static_v = ""
    initialization = ""
    line_count = -1
    print_result =""

    # Main 2) Load Template

    # Open file
    batch_normal = open("../Template/Function/BatchNormalization.txt")
    #with relu
    if argv[2] == "True":
        conv_s = open("../Template/Function/Conv2D_same_relu.txt")
    #without relu
    else :
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
    i_conv = open("../Template/Init/Conv_var_Initializer_f.txt")
    i_dense = open("../Template/Init/Dense_var_Initializer_f.txt")
    i_input = open("../Template/Init/Input_var_Initializer_f.txt")
    m = open("../Template/Main/C_Verification.txt")
    o1 = open("../Template/Print/Print_Output3D.txt")
    o2 = open("../Template/Print/Print_Output1D.txt")

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
    ZeroPadding2D = Template(zp.read())
    Init_conv = Template(i_conv.read())
    Init_dense = Template(i_dense.read())
    Init_input = Template(i_input.read())
    main = Template(m.read())
    output3d = Template(o1.read())
    output1d = Template(o2.read())

    # Main 3) Read Layer Information from CSV

    csv_file = open(argv[1])
    csv_reader = csv.DictReader(csv_file)

     # Main 4) Generate Function depending on layer_type

    for row in csv_reader:
        #Count Line number
        line_count+= 1
        line_str = str(line_count)
        #Get Input, Output shape
        input_shape = row["batch_input_shape"][1 : -1].split(", ")
        output_shape = row["batch_output_shape"][1 : -1].split(", ")

        # layer_type = convolution2D(I,O,B,W) (padding option: valid , same)
        if row["layer_type"] == 'Conv2D' :
            filter_shape = row["kernel_size"][1:-1].split(", ")
            stride_shape = row["strides"][1:-1].split(", ")
            l = {'Name' : row["name"], 'Input_channel' : input_shape[3], 'Input_width' : input_shape[1],'Stride_width': stride_shape[0],'Stride_height':stride_shape[1],
                 'Input_height' : input_shape[2], 'Output_channel' : output_shape[3],'Filter_width' : filter_shape[0],
                 'Filter_height' : filter_shape[1], 'Output_width' : output_shape[1], 'Output_height' : output_shape[2]}
            #Shared_variables(W,B)
            Shared_static_v += "static DATA_T W"+line_str+"[" + output_shape[3] + "][" + input_shape[3] + "][" + filter_shape[0] + "][" + filter_shape[1] + "];\n\t";
            Shared_static_v += "static DATA_T B"+line_str + "[" + output_shape[3] + "];\n\t";
            #SW_static_variables(O)
            SW_static_v += "static DATA_T O"+line_str+ "_SW[" + output_shape[3] + "][" + output_shape[1] + "][" + output_shape[2] + "];\n\t"
            #function def (Padding option)
            if row["padding"] == 'valid' :
                SW_def_func += Conv2D_valid.substitute(l) +"\n"
            else :
                SW_def_func += Conv2D_same.substitute(l) +"\n"
            m = {'Input_channel' : input_shape[3],'Output_channel' : output_shape[3],'Filter_width' : filter_shape[0],'Filter_height' : filter_shape[1],'line_number' : line_count}
            #Initialization
            initialization += Init_conv.substitute(m) + "\n\t"
            #Report Status
            SW_functions += "printf(\"[C_verifier.cpp]Calculate Conv2D"+line_str+"\\n\\n\");\n\t"
            #Function use
            SW_functions += "SW_" + row["name"]+ "(O" +str(line_count-1) +"_SW,O"+line_str +  "_SW,B" +line_str + ",W"+line_str +");\n\t"
            #print result
            c = {'Name':"Convolution2D",'Output_channel':output_shape[3],'Output_width':output_shape[1],'Output_height':output_shape[2],'line_number':line_count}
            print_result += output3d.substitute(c)+"\n"

        # layer_type = BatchNormalization(I, O)
        # if Batch size = None(one Input), Mean = 0, Var = 1 fixed.
        elif row["layer_type"] == 'BatchNormalization' :
            l = {'Name' : row["name"], 'Input_channel' : input_shape[3], 'Input_width' : input_shape[1],
                 'Input_height' : input_shape[2], 'Output_channel' : output_shape[3],
                  'Output_width' : output_shape[1], 'Output_height' : output_shape[2]}
            #SW_static_variables (O)
            SW_static_v += "static DATA_T O"+line_str+ "_SW[" + output_shape[3] + "][" + output_shape[1] + "][" + output_shape[2] + "];\n\t"
            #function def
            SW_def_func += BatchNormalization.substitute(l) +"\n"
            #Report Status
            SW_functions += "printf(\"[C_verifier.cpp]Calculate BatchNormalization"+line_str+"\\n\\n\");\n\t"
            #Function use
            SW_functions += "SW_" + row["name"]+ "(O" +str(line_count-1) +"_SW,O"+line_str +  "_SW);\n\t"
            #print result
            c = {'Name':"BatchNormalization",'Output_channel':output_shape[3],'Output_width':output_shape[1],'Output_height':output_shape[2],'line_number':line_count}
            print_result += output3d.substitute(c)+"\n"

        # layer_type = Activation(Relu)(I, O)
        elif row["layer_type"] == 'Activation' :
            l = {'Name' : row["name"], 'Input_channel' : input_shape[3], 'Input_width' : input_shape[1],
                 'Input_height' : input_shape[2], 'Output_channel' : output_shape[3],
                  'Output_width' : output_shape[1], 'Output_height' : output_shape[2]}
            #SW_static_variables (O)
            SW_static_v += "static DATA_T O"+line_str+ "_SW[" + output_shape[3] + "][" + output_shape[1] + "][" + output_shape[2] + "];\n\t"
            #function def
            SW_def_func += Relu.substitute(l) +"\n"
            #Report Status
            SW_functions += "printf(\"[C_verifier.cpp]Calculate Activation(Relu)"+line_str+"\\n\\n\");\n\t"
            #Function use
            SW_functions += "SW_" + row["name"]+ "(O" +str(line_count-1) +"_SW,O"+line_str+  "_SW);\n\t"
            #print result
            c = {'Name':"Activations.Relu",'Output_channel':output_shape[3],'Output_width':output_shape[1],'Output_height':output_shape[2],'line_number':line_count}
            print_result += output3d.substitute(c)+"\n"

        # layer_type = MaxPooling2D (I, O)
        elif row["layer_type"] == 'MaxPooling2D' :
            pool_shape = row["pool_size"][1:-1].split(", ")
            stride_shape = row["strides"][1:-1].split(", ")
            l = {'Name' : row["name"], 'Input_channel' : input_shape[3], 'Input_width' : input_shape[1],
                 'Input_height' : input_shape[2], 'Output_channel' : output_shape[3],
                  'Output_width' : output_shape[1], 'Output_height' : output_shape[2], 'Stride_width' : stride_shape[0],
                'Stride_height':stride_shape[1], 'Pool_width' : pool_shape[0], 'Pool_height' : pool_shape[1]}
            #SW_static_variables (O)
            SW_static_v += "static DATA_T O"+line_str+ "_SW[" + output_shape[3] + "][" + output_shape[1] + "][" + output_shape[2] + "];\n\t"
            #function def
            SW_def_func += MaxPooling2D.substitute(l) +"\n"
            #Report Status
            SW_functions += "printf(\"[C_verifier.cpp]Calculate MaxPooling2D"+line_str+"\\n\\n\");\n\t"
            #Function use
            SW_functions += "SW_" + row["name"]+ "(O" +str(line_count-1) +"_SW,O"+line_str+  "_SW);\n\t"
            #print result
            c = {'Name':"MaxPooling2D",'Output_channel':output_shape[3],'Output_width':output_shape[1],'Output_height':output_shape[2],'line_number':line_count}
            print_result += output3d.substitute(c)+"\n"

        # layer_type = AveragePooling2D (I, O)
        elif row["layer_type"] == 'AveragePooling2D' :
            pool_shape = row["pool_size"][1:-1].split(", ")
            stride_shape = row["strides"][1:-1].split(", ")
            l = {'Name' : row["name"], 'Input_channel' : input_shape[3], 'Input_width' : input_shape[1],
                 'Input_height' : input_shape[2], 'Output_channel' : output_shape[3],
                  'Output_width' : output_shape[1], 'Output_height' : output_shape[2], 'Stride_width' : stride_shape[0],
                'Stride_height':stride_shape[1], 'Pool_width' : pool_shape[0], 'Pool_height' : pool_shape[1]}
            #SW_static_variables (O)
            SW_static_v += "static DATA_T O"+line_str+ "_SW[" + output_shape[3] + "][" + output_shape[1] + "][" + output_shape[2] + "];\n\t"
            #function def
            SW_def_func += AveragePooling2D.substitute(l) +"\n"
            #Report Status
            SW_functions += "printf(\"[C_verifier.cpp]Calculate AveragePooling2D"+line_str+"\\n\\n\");\n\t"
            #Function use
            SW_functions += "SW_" + row["name"]+ "(O" +str(line_count-1) +"_SW,O"+line_str +  "_SW);\n\t"
            #print result
            c = {'Name':"AveragePooling2D",'Output_channel':output_shape[3],'Output_width':output_shape[1],'Output_height':output_shape[2],'line_number':line_count}
            print_result += output3d.substitute(c)+"\n"

        # layer_type = Add(I1,I2,O)
        elif row["layer_type"] == 'Add' :
            l = {'Name' : row['name'], 'Input_channel1' : output_shape[3], 'Input_width1' : output_shape[1],
                 'Input_height1' : output_shape[2],'Input_channel2' : output_shape[3], 'Input_width2' : output_shape[1],
                 'Input_height2' : output_shape[2],'Output_channel' : output_shape[3], 'Output_width' : output_shape[1],
                 'Output_height' : output_shape[2]}
            #SW_static_variables (O)
            SW_static_v += "static DATA_T O"+line_str+ "_SW[" + output_shape[3] + "][" + output_shape[1] + "][" + output_shape[2] + "];\n\t";
            #function def
            SW_def_func += Add.substitute(l) +"\n"
            #Report Status
            SW_functions += "printf(\"[C_verifier.cpp]Calculate Add"+line_str+"\\n\\n\");\n\t"
            #Function use
            SW_functions += "SW_" + row["name"]+ "(O" +str(line_count-1) +"_SW,O"+str(line_count-11) +  "_SW,O" +line_str+"_SW);\n\t"
            #print result
            c = {'Name':"Add",'Output_channel':output_shape[3],'Output_width':output_shape[1],'Output_height':output_shape[2],'line_number':line_count}
            print_result += output3d.substitute(c)+"\n"

        # layer_type = ZeroPadding2D(I,O)
        elif row["layer_type"] =="ZeroPadding2D":
            padding_shape = row["padding"][1:-1].split(", (")
            padding = padding_shape[0][1:-1].split(", ")
            l = {'Name' : row["name"], 'Input_channel' : input_shape[3], 'Input_width' : input_shape[1],
                 'Input_height' : input_shape[2], 'Output_channel' : output_shape[3],
                  'Output_width' : output_shape[1], 'Output_height' : output_shape[2], 'Padding_size': padding[0]}
            #SW_static_variables (O)
            SW_static_v += "static DATA_T O"+line_str+ "_SW[" + output_shape[3] + "][" + output_shape[1] + "][" + output_shape[2] + "];\n\t"
            #function def
            SW_def_func += ZeroPadding2D.substitute(l) +"\n"
            #Report Status
            SW_functions += "printf(\"[C_verifier.cpp]Calculate ZeroPadding2D"+line_str+"\\n\\n\");\n\t"
            #Function use
            SW_functions += "SW_" + row["name"]+ "(O" +str(line_count-1) +"_SW,O"+str(line_count) +  "_SW);\n\t"
            #print result
            c = {'Name':"ZeroPadding2D",'Output_channel':output_shape[3],'Output_width':output_shape[1],'Output_height':output_shape[2],'line_number':line_count}
            print_result += output3d.substitute(c)+"\n"

        # layer_type = InputLayer
        elif row["layer_type"] =="InputLayer":
            #Report Status
            SW_functions += "printf(\"[C_verifier.cpp]InputLayer\\n\\n\");\n\t"
            #SW_static_variables (I,O0)
            Shared_static_v += "static DATA_T I[" + input_shape[3] + "][" + input_shape[1] + "][" + input_shape[2] + "];\n\t";
            SW_static_v += "static DATA_T O0_SW[" + input_shape[3] + "][" + input_shape[1] + "][" + input_shape[2] + "];\n\t";
            m = {'Input_channel':input_shape[3],'Input_width':input_shape[1],'Input_height':input_shape[2]}
            #Initialization
            initialization += Init_input.substitute(m) + "\n\t"
            #print result
            c = {'Name':"InputLayer",'Output_channel':output_shape[3],'Output_width':output_shape[1],'Output_height':output_shape[2],'line_number':line_count}
            print_result += output3d.substitute(c)+"\n"

        # layer_type = Flatten(I,O)
        elif row["layer_type"] == "Flatten":
            l = {'Name':row["name"],'Input_channel':input_shape[3],'Input_width':input_shape[1],'Input_height':input_shape[2],'Output_channel':output_shape[1]}
            #SW_static_variables (O)
            SW_static_v += "static DATA_T O"+line_str+ "_SW[" + output_shape[1] + "];\n\t"
            #function def
            SW_def_func += Flatten.substitute(l) + "\n"
            #Report Status
            SW_functions += "printf(\"[C_verifier.py]Calculate Flatten"+line_str+"\\n\\n\");\n\t"
            #Function use
            SW_functions += "SW_"+row["name"]+"(O"+str(line_count-1)+"_SW,O"+line_str+"_SW);\n\t"
            #print result
            c = {'Name':"Flatten",'Output_channel':output_shape[1],'line_number':line_count}
            print_result += output1d.substitute(c)+"\n"

        # layer_type = Dense(I,W,B,O) (Activation option : relu , softmax)
        elif row["layer_type"] == "Dense":
            l = {'Name':row["name"],'Input_channel':input_shape[1],'Output_channel':output_shape[1]}
            #Shared_static_variable (B,W)
            Shared_static_v += "static DATA_T B"+line_str+ "[" + output_shape[1] + "];\n\t"
            Shared_static_v += "static DATA_T W"+line_str+ "[" + output_shape[1] + "][" + input_shape[1] + "];\n\t";
            #function def
            if row["activation"] == 'relu' : # Activation = relu
                SW_def_func += Dense_relu.substitute(l) + "\n"
            else :  # Activation = softmax
                SW_def_func += Dense_softmax.substitute(l) + "\n"
            m = {'Input_channel' : input_shape[1],'Output_channel' : output_shape[1],'line_number' : line_count}
            #Initialization
            initialization += Init_dense.substitute(m) + "\n\t"
            #SW_static_variable (O)
            SW_static_v += "static DATA_T O"+line_str+ "_SW[" + output_shape[1] + "];\n\t"
            #Report Status
            SW_functions += "printf(\"[C_verifier.cpp]Calculate Dense"+line_str+"\\n\\n\");\n\t"
            #functipn use
            SW_functions += "SW_"+row["name"]+"(O"+str(line_count-1)+"_SW,W"+line_str+",B"+line_str+",O"+line_str+"_SW);\n\t"
            #print result
            c = {'Name':"Dense",'Output_channel':output_shape[1],'line_number':line_count}
            print_result += output1d.substitute(c)+"\n"
        else :
            print ('Not defined')

    # Generate C file
    f = {'def_SW_functions':SW_def_func,'SW_static_variables' :SW_static_v,'Shared_static_variables':Shared_static_v,'SW_functions' :SW_functions,
         'Initialization':initialization, 'result' : print_result }
    c_file = main.substitute(f) + "\n";
    file = open('C_Verifier.cpp','w')
    file.write(c_file)
    file.close()
