from Sweeper import SweepTask, Sweeper
import os
from os.path import join
import json

TEST_NAME = 'vgg'

class ParamHolder:
    def __init__(self, key, params):
        self.params = params
        self.key = key
        self.curser = 0
    
    def getParam(self):
        return self.params[self.curser]

    def getKey(self):
        return self.key

    def incCurser(self):
        self.curser = (self.curser + 1)
        if(self.curser >= len(self.params)):
            self.curser = 0
            return 1
        return 0

def replaceInFile(params, inDir, outDir, inPath, outPath):
    with open(inPath, 'r') as inf:
        with open(outPath, 'w') as outf:
            red = inf.read();
            for param in params:
                red = red.replace('{'+param.getKey()+'}', str(param.getParam()))
            red = red.replace('{IN_DIR}', inDir)
            red = red.replace('{OUT_DIR}', outDir)
            outf.write(red)

def dumpParams(params, dumpPath):
    with open(dumpPath, 'w') as df:
        data={}
        for param in params:
            data[param.getKey()] = param.getParam()
        json.dump(data, df)

def replaceAndSaveSetting(params, index):
    inputDir = os.getcwd()
    outputDir = './tests/num'+str(index)
    os.system("mkdir -p "+outputDir+'/outputs')
    replaceInFile(params, inputDir, outputDir, TEST_NAME+'-template.cfg', join(outputDir, TEST_NAME+'.cfg'))
    return outputDir

class NHS(SweepTask):
    params = [
        ParamHolder('B1_L1_UF', [1, 2, 4, 8, 16, 32, 64]),
        ParamHolder('B1_L2_UF', [1, 2, 4, 8, 16, 32, 64]),
        ParamHolder('B2_L1_UF', [1, 2, 4, 8, 16, 32, 64,128]),
        ParamHolder('B2_L2_UF', [1, 2, 4, 8, 16, 32, 64,128]),
        ParamHolder('B3_L1_UF', [1, 2, 4, 8, 16, 32, 64,128,256])
            ] 
    containerIndex = 0

    def prepareSetting(self):
        cont = True
        for index in range(len(NHS.params)):
            if(NHS.params[index].incCurser()):
                if(index >= len(NHS.params)-1):
                    cont = False
                    break
            else:
                break
        self.container = replaceAndSaveSetting(NHS.params, NHS.containerIndex)
        NHS.containerIndex += 1
        return cont

    def doThings(self):
        #os.system("sh "+self.container+"/run.sh")
        print("done test: ", self.container)

if __name__ == '__main__':
    sweeper = Sweeper(NHS)
    sweeper.sweep()



