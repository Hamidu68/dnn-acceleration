from ..Models import *
from ..Layers import *
from .CodeGenerators import CodeGenerators


class HW_test(CodeGenerators):
    def __init__(self, models=[], dtype='DATA_T',name=''):
        super().__init__(models, dtype)
        self.model_sw = models[0]
        self.model_hw = models[1]
        self.dtype = dtype;
        self.model_name = name;

    def generate(self):
    	#o1 = open("../Template/Main/vivado_cpp.txt")
    	#o2 = open("../Template/Main/vivado_test_cpp.txt")
    	#f1 = open('../../Produced_code/'+model_name+'/'+model_name+'.cpp', 'w')
    	#f2 = open('../../Produced_code/'+model_name+'/'+model_name+'_test.cpp', 'w')
        return
'''
        f1.write(o1.read().format(Data_type = dtype,
        	hw_static_variables=self.gen_hw_static_variables(),
        	Stream_io=,
        	hw_def_layer=self.gen_hw_def_layer(),
        	sw_def_layer=self.gen_sw_def_layer(),
        	assign_sw2hw=,
        	assign_hw2sw=,
        	hw_func_params=,
        	sw_func_params=,
        	model_name=self.model_name,
        	call_params=,
        	optimized_code=,
        	hw_output_streams=,
        	hw_call_layer=gen_hw_call_layer(),
        	output_stream=,
        	sw_output_variables=sefl.gen_sw_output_variables(),
        	sw_call_layer=self.gen_sw_call_layer()))

        f2.write(o2.read().format(Data_type= dtype,
        	model_name=self.model_name,
        	hw_func_params=,
        	sw_func_params=,
        	sw_static_variables=self.gen_sw_static_variables(),
        	Initialization=self.gen_initialization(),
        	call_params= ))
        '''