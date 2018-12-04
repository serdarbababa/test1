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

    def genInstantSymbol(self, verbose =False):
        x = self.symbols[ random.randint(0, 15)]
        if verbose: print(x)
        return x
    ################################################
    def genSample(self, signalCount, verbose=False):
        if verbose: print("generate sample data")
        signals = []
        for i in range(signalCount):
            a = self.genData(["normal", 100, 100, 8])
            # print(a)
            sig = []
            for j in range(8):
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
        return coefs

    ################################################
    def generateOperationsSymbols(self, operations_count, Test=False, verbose=False):
        ops_ids = []
        symbolSet = []

        for i in range(operations_count):
            if (verbose): print(i, end="\t")
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
        sig = []
        if (verbose):
            print(input_data)
        for j in range(int(len(input_data))):
            output = 7
            for k in range(7):
                if (input_data[j] < borders[k]):
                    output = k
                    break
            if verbose:
                print(output, end=" ")
            sig.append(output)
        if verbose:
            print()
        return sig

    ################################################
    def generateInputData(self, op_count, verbose=False):
        for i in range(10):
            encoded, symbol_based = self.generateOperationsSymbols(op_count, False, verbose)
            if len(symbol_based)>0:
                break
        if (verbose):
            print("operation symbols")
            [print(i, symbol_based[i]) for i in range(len(symbol_based))]
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
        [print(i, "\t\t", symbols_correspondence[i], "\t\t", self.symbols[i]) for i in range(len(self.symbols))]

    ################################################
    def __init__(self):
        self.symbols = self.genSample(16)


