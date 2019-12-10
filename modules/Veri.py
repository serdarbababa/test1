import numpy as np
from pywt import wavedec
import matplotlib.pyplot as plt
from networkx.drawing.nx_agraph import graphviz_layout
import networkx as nx
import pandas as pd
import seaborn as sns
import random


class Veri():
    ################################################
    def genData(self, param, show=False):
        #print(param)
        a = []
        if param[0] == "normal":
            mu, sigma, s = param[1], param[2], param[3]
            a = np.random.normal(mu, sigma, size=s)
        elif param[0] == 'uniform':
            mi, ma, s = param[1], param[2], param[3]
            a = np.random.uniform(mi, ma, s)
        elif param[0] == "poisson":
            rate, s = param[1], param[2]
            a = np.random.poisson(rate, s)
        if (show):
            count, bins, ignored = plt.hist(s, 14, density=True)
        return a

    def genInstantSymbol(self, verbose=False):
        x = self.symbols[random.randint(0, 15)]
        # x = self.symbols[random.randint(0, 15)]
        if verbose: print(x)
        return x

    ##################### new functions#########################33
    def predefinedSymbols(self):
        sample = {}
        sample["kare"] = {"p": 0.2, "val": [0, 1, 1, 0]}
        sample["ucgen"] = {"p": 0.1, "val": [0, 0.25, 0.5, 0.75, 1, 0.75, 0.5, 0.25]}
        sample["null"] = {"p": 0.3, "val": [0, 0, 0, 0]}
        sample["sin"] = {"p": 0.3, "val": [0, 0.5, 1, 0.5, 0, -0.5, -1, -0.5]}
        sample["cos"] = {"p": 0.1, "val":   [1, 0.5, 0, -0.5, -1, -0.5, 0, 0.5]}
        return sample

    def genSignalWithSimpleSymbols(self, symbolCountInSignal=100, add_nulls = False, verbose = False):
        self.symbols = self.predefinedSymbols()
        #print(len(self.symbols))
        names = list(self.symbols.keys())
        samples = self.genData(["uniform", 0, len(self.symbols), symbolCountInSignal]).astype(int)
        #print("sample = ",samples)
        signal = []
        for s in samples:
            if verbose:
                print( names[s],self.symbols[names[s]]["val"])
            signal = signal + self.symbols[names[s]]["val"]
            if add_nulls:
                signal = signal + self.symbols["null"]["val"]
        return signal

    ################################################
    def genSample(self, signalCount, verbose=False):
        if verbose: print("generate sample data")
        signals = []
        for i in range(signalCount):
            a = self.genData(["normal", 100, 100, self.sample_len])
            # print(a)
            sig = []
            for j in range(self.sample_len):
                sig.append(int(a[j]))
            signals.append(sig)
        for i in range(signalCount):
            if verbose: print(signals[i])
        return signals

    def genVarLenSample(self, signalCount, verbose=False):
        if verbose: print("generate sample data")
        signals = []
        for i in range(signalCount):
            sample_length = random.randint(self.sample_len, self.sample_len * 3)
            a = self.genData(["normal", 100, 100, sample_length])
            # print(a)
            sig = []
            for j in range(sample_length):
                sig.append(int(a[j]))
            signals.append(sig)
        for i in range(signalCount):
            if verbose: print(signals[i])
        return signals

    ################################################
    def mergeList(self, input_data, verbose=False):
        if verbose:
            print("merge data")
        merged_list = []
        for l in input_data:
            merged_list += list(l)
        return merged_list

    ################################################
    def listToPandasDF(self, input_data):
        df = pd.DataFrame(input_data)
        return df

    ################################################
    def getWaveletCoefs(self, input_data):
        coefs = []

        girdi = np.array(input_data)  # np.array([1,2,3,4,5,6,7,8])*1
        coeff = wavedec(girdi, 'haar', level=int(np.log2(len(girdi))))
        coefs = (self.mergeList(coeff))
        coefs = np.round(coefs, decimals=2)
        return coefs

    ################################################
    def generateOperationsSymbols(self, operations_count, Test=False, verbose=False):
        ops_ids = []
        symbolSet = []

        for i in range(operations_count):
            if (verbose): print(i, "\t", )
            a = self.genData(["uniform", 0, 10, 4])
            a = [int(x) for x in a]
            if (Test):
                a[3] = 0
            go = True
            # a=[4, 5, 9, -5]

            rez = 0
            if (int(a[1]) % 4 == 0):  # operation is +
                if (a[0] + a[2] > 9):
                    go = False
                else:
                    rez = a[0] + a[2]
            elif (int(a[1]) % 4 == 1):  # operation is -
                if a[0] < a[2]:
                    go = False
                else:
                    # print("here", a[0] - a[2] , a[0] > a[2])
                    rez = a[0] - a[2]
            elif (int(a[1]) % 4 == 2):  # operation is *
                if (a[0] * a[2] > 9):
                    go = False
                else:
                    rez = a[0] * a[2]
            elif (int(a[1]) % 4 == 3):  # operation is -
                if (a[2] == 0):
                    go = False
                else:
                    rez = int(a[0] / a[2])
            # rint(go)
            if go:
                if verbose: print(go, rez)
                a[3] = rez
                ops_ids.append(a)
                symbolSet.append(self.symbols[a[0]])
                symbolSet.append(self.symbols[a[1] % 4 + 10])
                symbolSet.append(self.symbols[a[2]])
                symbolSet.append(self.symbols[14])
                if (not Test):
                    symbolSet.append(self.symbols[a[3]])
                else:
                    symbolSet.append(self.symbols[15])
        return ops_ids, symbolSet

    ################################################
    def quantize(self, input_data, len_of_data, verbose=False):
        borders = [-200, -100, -50, 0, 50, 100, 200]
        borders = [-6000, -4000, -2000, -1000, -500, -100, -50, 0, 50, 100, 500, 1000, 2000, 4000, 6000]
        # borders = [-100, -50, -30, -20, -10, -5, -3, 0, 3, 5, 10, 20, 30, 50, 100]
        sig = []
        if (verbose):
            print(input_data)
        for j in range(int(len(input_data))):
            output = len(borders)
            for k in range(len(borders)):
                if (input_data[j] < borders[k]):
                    output = k
                    break
            if verbose:
                print(output, " ", )
            sig.append(output)
        if verbose:
            print()
        return sig

    ################################################
    def generateInputData(self, op_count, verbose=False):
        for i in range(10):
            encoded, symbol_based = self.generateOperationsSymbols(op_count, False, verbose)
            if len(symbol_based) > 0:
                break
        if (verbose):
            print("operation symbols")
            for i in range(len(symbol_based)):
                print(i, symbol_based[i])
            print()
        return symbol_based

    ################################################
    def addNoise(self, data, noise_mean, noise_std):
        return data

    ################################################
    def rmse(self, predictions, targets):
        return np.sqrt(((predictions - targets) ** 2).mean())

    ################################################
    def displaySymbols(self):
        symbols_correspondence = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "+", "-", "*", "/", "=", "?"]
        print("indice", "symbol", "pattern")
        for i in range(len(self.symbols)):
            print(i, symbols_correspondence[i], self.symbols[i])

            ################################################

    def __init__(self, sample_len=8):
        self.sample_len =""# sample_len
        self.symbols =""# self.genSample(16)
        # self.symbols = self.genVarLenSample(16)
