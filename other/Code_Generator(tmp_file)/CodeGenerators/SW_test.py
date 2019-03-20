from ..Models import *
from ..Layers import *
from .CodeGenerators import *
import os


class SW_test(CodeGenerators):

    def __init__(self, models=[], dtype='DATA_T',name=''):
        super(SW_test, self).__init__(models, dtype)
        self.model_sw = models[0]
        self.model_name = name

    def generate(self):
        o1 = open("Code_Generator/Template/Main/main_sw.txt")
        output_path_1 = "\"Produced_code/" + self.model_name +"/Output/c_output.txt\""
        output_path_2 = "\"Produced_code/" + self.model_name +"/Output/c_output_num.txt\""
        file = open('Produced_code/'+self.model_name+'/C_verifier.cpp', 'w')
        file.write(o1.read().format(sw_def_layer=self.gen_sw_def_layer(),
                                              sw_static_variables=self.gen_sw_static_variables(),
                                              sw_output_variables=self.gen_sw_output_variables(),
                                              Initialization=self.gen_initialization(),
                                              sw_call_layer=self.gen_sw_call_layer(),
                                              result=self.gen_print_result(),
                                              output_path_1 = output_path_1,
                                              output_path_2 = output_path_2
                                              ))
        file.close()
        return 'Produced_code/'+self.model_name+'/C_verifier.cpp';
