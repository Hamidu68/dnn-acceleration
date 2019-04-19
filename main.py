import os, sys
import json

from src import *

if __name__ == '__main__':

    #set configs
    with open('config.json', 'r') as fr:
        jData = json.load(fr)
        extract_config = jData["extract_configs"]
        model_info = jData["model_info"]
        model_name = jData["model"]
        run_c_test = jData["sw_test"]
        run_vivado_test = jData["vivado_test"]
        dtype = jData["data_type"]
        batch = jData["batch"]
        random_range = int(jData["random_range"])
        weight_file_path = jData["weight_file_path"]
        input_file_path = jData["input_file_path"]
        use_trained_weight= jData["use_trained_weight"]
        trained_weight_file_path= jData["trained_weight_file_path"]
        image_file_path= jData["image_file_path"]
        output_path = jData["output_path"] + model_name
        template_path = jData["template_path"]

    paths = []
    paths.append(output_path)
    paths.append(template_path)

    #extract model layer information. ex) vgg19.csv
    if extract_config == "True":
        extract_configs(model_name, model_info)

    #run keras vs c test with trained weights and given input
    if run_c_test == "True" and use_trained_weight == "True":
        path.append(trained_weight_file_path)
        path.append(image_file)
        sw_test(model_info, model_name, dtype, batch, paths)

    #run keras vs c test with random weights and input
    if run_c_test == "True" and use_trained_weight == "False":
        paths.append(weight_file_path)
        paths.append(input_file_path)
        variable_generator(model_info, weight_file_path, input_file_path, random_range, dtype)
        sw_test(model_info, model_name, dtype, batch, paths)

    #generate vivado code
    if run_vivado_test == "True" :
        vivado_test(model_info, model_name, dtype, batch,paths)
