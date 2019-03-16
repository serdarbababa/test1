import numpy as np
from pywt import wavedec
import matplotlib.pyplot as plt
from networkx.drawing.nx_agraph import graphviz_layout
import networkx as nx
import pandas as pd
import seaborn as sns
import random
from modules.Veri import Veri
import matplotlib.pyplot as plt
import networkx as nx




class BaseStructure:
        ################################################
        def addBranch(self, input_data):
            poz = 0
            for j in range(len(input_data)): # for all data
                d = input_data[j]

                nei = list(self.agac.neighbors(poz)) # get neighbours of node with id poz
                if len(nei) == 0: # if there is no node, directly add node
                    self.agac.add_node(self.counter, value=d, occurance_count=1, id=-1)
                    self.agac.add_edge(poz, self.counter)
                    poz = self.counter
                    self.counter += 1
                else:
                    k = -1
                    for n in nei:
                        if (self.agac.node[n]['value'] == d):
                            k = n
                            break
                    if (k >= 0):
                        poz = k
                        self.agac.node[k]['occurance_count'] += 1
                    else:
                        self.agac.add_node(self.counter, value=d, occurance_count=1, id=-1)
                        self.agac.add_edge(poz, self.counter)
                        poz = self.counter
                        self.counter += 1
            return poz
        ################################################
        def checkBranch(self, input_data):
            poz = 0
            for j in range(len(input_data)): # for all data
                d = input_data[j]

                nei = list(self.agac.neighbors(poz)) # get neighbours of node with id poz
                if len(nei) == 0: # if there is no node, directly add node
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
        ################################################
        def getBranchGivenStartNodeValue(self, startNodeValue):
            data = []
            nei = list(self.agac.neighbors(0))
            # print(nei)
            k = -1
            for n in nei:
                if (self.agac.node[n]['value'] == startNodeValue):
                    k = n
                    break
            # print(k)
            # data.append(k)

            while (k >= 0):
                nei = list(self.agac.neighbors(k))
                if len(nei) == 0:
                    k = -1
                else:
                    k = nei[0]
                    data.append( round(self.agac.node[k]['value'],2))
            return data
        ################################################
        def getBranchIDsGivenStartNodeValue(self, startNodeValue):
            data = []
            nei = list(self.agac.neighbors(0))
            # print(nei)
            k = -1
            for n in nei:
                if (self.agac.node[n]['value'] == startNodeValue):
                    k = n
                    break
            # print(k)
            # data.append(k)

            while (k >= 0):
                nei = list(self.agac.neighbors(k))
                if len(nei) == 0:
                    k = -1
                else:
                    k = nei[0]
                    # data.append(GG.node[k]['value'])
                    data.append(k)
            return data
        ################################################

        def agCizdir(self,title = "Tree structure", short=False):
            plt.rcParams['figure.figsize'] = [15, 10]
            #labels = dict((n, round(d['value'], 2)) for n, d in self.agac.nodes(data=True))
            labels = dict((n, d['value']) for n, d in self.agac.nodes(data=True))
            # pos=nx.graphviz_layout(GG, prog='dot')
            #pos = graphviz_layout(self.agac, prog='dot')
            # nx.spring_layout(GG)
            pos = nx.spring_layout(self.agac)

            plt.title(title +" node values")
            nx.draw_networkx(self.agac, pos=pos, arrows=True, with_labels=True, labels=labels)
            plt.show()

        def plotGraph(self,title = "Tree structure", short=False):
            plt.rcParams['figure.figsize'] = [15, 10]
            #labels = dict((n, round(d['value'], 2)) for n, d in self.agac.nodes(data=True))
            labels = dict((n, d['value']) for n, d in self.agac.nodes(data=True))
            # pos=nx.graphviz_layout(GG, prog='dot')
            pos = graphviz_layout(self.agac, prog='dot')
            # nx.spring_layout(GG)

            plt.title(title +" node values")
            nx.draw_networkx(self.agac, pos=pos, arrows=True, with_labels=True, labels=labels)
            plt.show()
            if (short):
                return
            plt.title("node ids")
            nx.draw_networkx(self.agac, pos=pos, arrows=True, with_labels=True)
            plt.show()

            plt.title("node frequency")
            labels = dict((n, d['occurance_count']) for n, d in self.agac.nodes(data=True))
            nx.draw_networkx(self.agac, pos=pos, arrows=True, with_labels=True, labels=labels)
            plt.show()

            plt.title("final nodes ids")
            labels = dict((n, d['id']) for n, d in self.agac.nodes(data=True))
            nx.draw_networkx(self.agac, pos=pos, arrows=True, with_labels=True, labels=labels)
            plt.show()

            ################################################
        ################################################
        def learnSymbol(self, raw_data, verbose):
            cbid = self.addBranch(raw_data)

            # abid = self.getBranchId(qdata)
            if (verbose):
                print("Leaf id: ", cbid)
            return cbid
        ################################################
        def __init__(self):
            self.agac = nx.DiGraph()
            self.agac.add_node(0, value=999999, occurance_count=1, id=-1)
            self.counter = 1
            self.v = Veri()
