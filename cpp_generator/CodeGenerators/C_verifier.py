from ..Models import *
from ..Layers import *
from .CodeGenerators import CodeGenerators

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
        f.write(template.format(inner='',
                                sw_def_layer = get_sw_def_layer()))
        return

    template = '''
#include <iostream>
using namespaces std;

int main() {
    {inner}
    return;
}
'''
