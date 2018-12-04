import numpy as np
from pywt import wavedec
import matplotlib.pyplot as plt
from networkx.drawing.nx_agraph import graphviz_layout
import networkx as nx
import pandas as pd
import seaborn as sns
import random

from Veri  import Veri 
from BaseStructure import BaseStructure 

class Abstract(BaseStructure):
    def __init__(self, windowLength, offset):
        self.windowLength =windowLength
        self.offset= offset
        BaseStructure.__init__(self)


    def learnSymbol(self, raw_data,verbose):
        qdata = self.preprocess(raw_data)

        if (verbose):
            print("preprocessed data: ", qdata)

        abid = self.addBranch(qdata)
        #abid = self.getBranchId(qdata)
        if (verbose):
            print("Abstract leaf id: ", abid)
        return abid

    ################################################
    def checkBranch(self, raw_data):
        input_data= self.preprocess(raw_data)
        poz = 0
        for j in range(len(input_data)):  # for all data
            d = input_data[j]

            nei = list(self.agac.neighbors(poz))  # get neighbours of node with id poz
            if len(nei) == 0:  # if there is no node, directly add node
                return None
            else:
                k = -1
                for n in nei:
                    if (self.agac.node[n]['value'] == d):
                        k = n
                        break
                if (k >= 0):
                    poz = k
                else:
                    return None
        return poz

    def preprocess(self, symbol_data):
        c = (self.v.getWaveletCoefs(symbol_data))
        # print(c)
        quantized_input = self.v.quantize(c, len_of_data=self.windowLength)
        return quantized_input

class Context1(BaseStructure):
    def __init__(self):
        BaseStructure.__init__(self)
        
        
class Context2(BaseStructure):
    def __init__(self):
        BaseStructure.__init__(self)

    def checkBranch(self, input_data):
        poz = 0
        for j in range(len(input_data)):  # for all data
            d = input_data[j]

            nei = list(self.agac.neighbors(poz))  # get neighbours of node with id poz
            if len(nei) == 0:  # if there is no node, directly add node
                return None
            else:
                k = -1
                for n in nei:
                    if (self.agac.node[n]['value'] == d):
                        k = n
                        break
                if (k >= 0):
                    poz = k
                else:
                    return None
        return self.agac.node[poz]['value']
    def checkShortBranch(self, input_data):
        poz = 0
        for j in range(len(input_data)):  # for all data
            d = input_data[j]

            nei = list(self.agac.neighbors(poz))  # get neighbours of node with id poz
            if len(nei) == 0:  # if there is no node, directly add node
                return None
            else:
                k = -1
                for n in nei:
                    if (self.agac.node[n]['value'] == d):
                        k = n
                        break
                if (k >= 0):
                    poz = k
                else:
                    return None
        nei = list(self.agac.neighbors(poz))
        poz = nei[0]

        return self.agac.node[poz]['value']

    
class Actuator(BaseStructure):
    def __init__(self, windowLength):
        self.windowLength = windowLength
        BaseStructure.__init__(self)
        self.learningRate = 0.1
        self.loopCount = 10

    def learnSymbol(self, raw_data,verbose):
        abid = self.checkBranch(raw_data)
        if not abid:
            sample_sound = self.v.mergeList([[raw_data], (self.v.genSample(1, verbose)[0])])
            abid = self.addBranch(sample_sound)
            if (verbose):
                print("Actuator leaf id: ", abid)
        return abid


    def checkBranch(self, input_data):
        poz = 0

        d = input_data

        nei = list(self.agac.neighbors(poz))  # get neighbours of node with id poz
        if len(nei) == 0:  # if there is no node, directly add node
            return None
        else:
            k = -1
            for n in nei:
                if (self.agac.node[n]['value'] == d):
                    k = n
                    break
            if (k >= 0):
                poz = k
            else:
                return None
        return poz

    def fineTune(self, branch_start_node, raw_data):
        nodes = self.getBranchIDsGivenStartNodeValue(branch_start_node)
        for j in range(self.loopCount):
            for i in range(len(nodes)):
                # print (i, branch[i] , GG.node[nodes[i]]['value'])
                self.agac.node[nodes[i]]['value'] += round(
                    (raw_data[i] - self.agac.node[nodes[i]]['value']) * self.learningRate, 2)

                


                
class Spektron:

    def __init__(self,abstractWindowLength=8 ):
        self.v = Veri()
        self.abstract = Abstract(abstractWindowLength , offset= 0)
        self.context1 = Context1()
        self.context2 = Context2()
        self.actuator = Actuator(windowLength = abstractWindowLength +1)
        self.ConnInputBuffer=[]
        self.ConnInputBufferSize = 5

    def displaySpektron(self):
        self.v.displaySymbols()
        self.abstract.plotGraph(title="Abstract", short=True)
        self.context1.plotGraph(title="Context1", short=True)
        self.context2.plotGraph(title="Context2", short=True)
        self.actuator.plotGraph(title="Actuator", short=True)
    def getInstantOperationInput(self, verbose = False):
        raw_input= self.v.generateInputData(1, verbose)

        raw_input = self.v.addNoise(raw_input, noise_mean=0, noise_std=0)
        if (verbose):
            print("gen instant input: ", raw_input)
        return raw_input

    def train_Abstract_Context1_Actuator(self, raw_data, verbose = False):
      print("")

    def oneBeat(self, symbol = None, verbose=False):
        if verbose: print("One Beat ", end="")
        if not symbol:
            symbol = self.v.genInstantSymbol(verbose)
        if verbose: print("Input Symbol = ", symbol)

        abstract_output = self.abstract.learnSymbol(symbol,verbose)

        if verbose: print("Abstract Branch ID : ",self.abstract.checkBranch(symbol ))

        context1_output = self.context1.learnSymbol([abstract_output], verbose)
        if verbose :print ("Context1 Branch ID : ",self.context1.checkBranch([abstract_output]))

        actuator_output = self.actuator.learnSymbol(context1_output, verbose)
        self.actuator.fineTune( context1_output, symbol)
        if verbose : print("Actuator Branch ID", self.actuator.checkBranch(context1_output))

    def oneComplexBeat(self, symbol,  getOutput= False , verbose=False):
        if verbose: print("One Complex Beat ",end="")

        #symbol = self.v.genInstantSymbol(verbose)
        if verbose: print("Input Symbol = " , symbol)

        abstract_output = self.abstract.learnSymbol(symbol,verbose)

        if verbose: print(self.abstract.checkBranch(symbol ))

        context1_output = self.context1.learnSymbol([abstract_output], verbose)
        if verbose :print (self.context1.checkBranch([abstract_output]))

        self.ConnInputBuffer.append(context1_output)
        if (len(self.ConnInputBuffer) <self.ConnInputBufferSize ):
            return None

        if getOutput :  print("Operation Context 2 Input ", self.ConnInputBuffer)

        context2_output = self.context2.addBranch(self.ConnInputBuffer)
        #dbid = self.getBranchId(self.ConnInputBuffer, self.ConnTree, self.counterAbstract, WL=5, overlap=0)
        context2_output_value = self.context2.checkBranch(self.ConnInputBuffer)
        #if verbose :print (self.context2.checkBranch(self.ConnInputBuffer))


        #actuator_output = self.actuator.learnSymbol(context2_output, verbose)
        if verbose : print(self.actuator.checkBranch(context2_output_value))
        self.ConnInputBuffer = []
        if getOutput : print("output branch values ",self.actuator.getBranchGivenStartNodeValue(context2_output_value))
        #print("output branch IDs ",self.actuator.getBranchIDsGivenStartNodeValue(context2_output))

    def checkOperation(self, symbol, getOutput= False , verbose=False):
        if verbose: print("One Complex Beat ",end="")

        #symbol = self.v.genInstantSymbol(verbose)
        if verbose: print("Input Symbol = " , symbol)

        abstract_output = self.abstract.learnSymbol(symbol,verbose)

        if verbose: print(self.abstract.checkBranch(symbol ))

        context1_output = self.context1.learnSymbol([abstract_output], verbose)
        if verbose :print (self.context1.checkBranch([abstract_output]))

        self.ConnInputBuffer.append(context1_output)
        if (len(self.ConnInputBuffer) <self.ConnInputBufferSize-1 ):
            return None

        if getOutput : print("Operation Context 2 Input ", self.ConnInputBuffer)

        context2_output = self.context2.addBranch(self.ConnInputBuffer)
        #dbid = self.getBranchId(self.ConnInputBuffer, self.ConnTree, self.counterAbstract, WL=5, overlap=0)
        context2_output_value = self.context2.checkShortBranch(self.ConnInputBuffer)
        #if verbose :print (self.context2.checkBranch(self.ConnInputBuffer))


        #actuator_output = self.actuator.learnSymbol(context2_output, verbose)
        if verbose : print(self.actuator.checkBranch(context2_output_value))
        self.ConnInputBuffer = []
        if getOutput : print("output branch values ",self.actuator.getBranchGivenStartNodeValue(context2_output_value))
        #print("output branch IDs ",self.actuator.getBranchIDsGivenStartNodeValue(context2_output))
