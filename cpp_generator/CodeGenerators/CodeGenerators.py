from ..Models import *
from ..Layers import *

class CodeGenerators():
    def __init__(models=[], dtype='DATA_T'):
        self.models = []
        self.dtype = dtype

    def gen_print_result():
        return
    def gen_Initialization():
        return
    
    def generate():
        return

class C_verifier(CodeGenerators):
    def __init__(models=[], dtype='DATA_T'):
        super().__init__(models, dtype)
        self.model_sw = models[0]

    def gen_sw_def_layer():
        return
    def gen_sw_static_variables():
        return
    def gen_sw_output_variables():
        return
    def gen_sw_call_layer():
        return
    
    def generate():
        return

class HW_test(CodeGenerators):
    def __init__(models=[], dtype='DATA_T'):
        super().__init__(models, dtype)
        self.model_sw = models[0]
        self.model_hw = models[1]

    def generate():
        return
        
class DAC2017_test(CodeGenerators):
    def __init__(models=[], dtype='DATA_T'):
        super().__init__(models, dtype)
        self.model_sw = models[0]
        self.model_dac2017 = models[1]

    def generate():
        return


