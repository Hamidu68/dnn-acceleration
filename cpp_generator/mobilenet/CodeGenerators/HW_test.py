from ..Models import *
from ..Layers import *
from .CodeGenerators import CodeGenerators


class HW_test(CodeGenerators):
    def __init__(self, models=[], dtype='DATA_T'):
        super().__init__(models, dtype)
        self.model_sw = models[0]
        self.model_hw = models[1]

    def generate(self):
        return