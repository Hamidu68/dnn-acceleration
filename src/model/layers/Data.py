class Data():
    def __init__(self, dtype='DATA_T', shape=(), name='W'):
        self.dtype = dtype
        self.shape = shape
        self.name = name

    def set_name(self,name=''):
        self.name = name

    def get_shape():
        return self.shape

    def get_static_variable(self):
    	if len(self.shape) == 1:
            return 'static {} {}[{}];\n\t'.format(self.dtype, self.name, self.shape[0])
        elif len(self.shape) == 2:
            return 'static {} {}[{}][{}];\n\t'.format(self.dtype, self.name, self.shape[0], self.shape[1])
        elif len(self.shape) == 3:
            return 'static {} {}[{}][{}][{}];\n\t'.format(self.dtype, self.name, self.shape[2], self.shape[0], self.shape[1])
        else :
            return 'static {} {}[{}][{}][{}][{}];\n\t'.format(self.dtype, self.name, self.shape[0], self.shape[1], self.shape[2], self.shape[3])

    def get_func_param(self):
    	if len(self.shape) == 1:
            return '{} {}[{}], '.format(self.dtype, self.name, self.shape[0])
        elif len(self.shape) == 2:
            return '{} {}[{}][{}],'.format(self.dtype, self.name, self.shape[0], self.shape[1])
        elif len(self.shape) == 3:
            return '{} {}[{}][{}][{}],'.format(self.dtype, self.name, self.shape[2], self.shape[1], self.shape[0])
        else :
            return '{} {}[{}][{}][{}][{}],'.format(self.dtype, self.name, self.shape[0], self.shape[1], self.shape[2], self.shape[3])


    def get_call_param(self):
    	return '{}, '.format(self.name)