import numpy as np
from pywt import wavedec
import matplotlib.pyplot as plt
from networkx.drawing.nx_agraph import graphviz_layout
import networkx as nx
import pandas as pd
import seaborn as sns
import random


class Data():
    def predefinedSymbols():
        sample = {}
        sample["kare"] = {"p": 0.2, "val": [0, 1, 1, 0]}
        sample["ucgen"] = {"p": 0.2, "val": [0, 0.25, 0.5, 0.75, 1, 0.75, 0.5, 0.25]}
        sample["null"] = {"p": 0.3, "val": [0, 0, 0, 0]}
        sample["sin"] = {"p": 0.3, "val": [0, 0.5, 1, 0.5, 0, -0.5, -1, -0.5]}
        return sample

    ############## data preparation ##################################
    def genData( param, show=False):
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

    def genSignal( sampleCount, verbose=False):
        if verbose: print("generate sample signal of ", sampleCount, "symbols")
        signals = []
        for i in range(sampleCount):
            a = genData(["normal", 100, 100, sampleCount])

            sig = []
            for j in range(sampleCount):
                sig.append(int(a[j]))
            signals.append(sig)
        for i in range(signalCount):
            if verbose: print(signals[i])
        return signals
    def genVarLenSample( signalCount, verbose=False):
        if verbose: print("generate sample data")
        signals = []
        for i in range(signalCount):
            sample_length = random.randint(self.sample_len, self.sample_len*3)
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
    def mergeList( input_data, verbose=False):
        if verbose:
            print("merge data")
        merged_list = []
        for l in input_data:
            merged_list += list(l)
        return merged_list