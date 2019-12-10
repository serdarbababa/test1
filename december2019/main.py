# TODO 1 genrate simple symbols/signal
# todo 2 generate more symbols/complex
# todo 3 generate comples symbols/signal

# todo get the signal and learn components of the sygmal
# todo get the learnt components and make more complex elements (variable size)
# todo get the complex elements and learn rules


from modules.Components import Veri
import numpy as np
import matplotlib.pyplot as plt
from numpy import dot
from numpy.linalg import norm


# veri= Veri()

class yapi:
    def getSymbols(self):
        print("Symbols:")
        symbols = self.veri.predefinedSymbols()
        for s in symbols.keys():
            print(s, symbols[s]["val"])
        print()

    def learn(self, dataa):
        data = list(dataa)
        max_sim = -1
        max_sim_id = -1

        if data in self.branches:
            max_sim_id = self.branches.index(data)
            max_sim = 1
        else:
            # print(type(data), type(np.zeros(len(data))))
            # if  np.array_equal(data, np.zeros(len(data))):
            if data == [0] * len(data):
                # print()
                # print("adding 0 ", data)
                # print()
                self.branches.insert(0, data)
                self.branche_freq.insert(0, 0)
                self.start_index = 1
                max_sim_id = 0
            else:
                for i, b in enumerate(self.branches):
                    if i >= self.start_index:
                        cos_sim = dot(b, data) / (norm(b) * norm(data))
                        if cos_sim > max_sim:
                            max_sim = cos_sim
                            max_sim_id = i
                        # print(i, cos_sim, max_sim_id)
        # print("maxsim , id, data|", data in self.branches , "|",max_sim, max_sim_id, dataa)
        if max_sim < 0.9:

            if not data in self.branches:
                # print("\nnew brabch added ", data)
                self.branches.append(data)
                self.branche_freq.append(0)
                max_sim_id = len(self.branches)
        else:
            self.branche_freq[max_sim_id] += 1
        return max_sim_id

    def test(self, dataa):
        data = list(dataa)
        max_sim = -1
        max_sim_id = -1

        if data in self.branches:
            max_sim_id = self.branches.index(data)
            max_sim = 1
        else:
            for i, b in enumerate(self.branches):
                # print(i,b,data)
                if i >= self.start_index:
                    cos_sim = dot(b, data) / (norm(b) * norm(data))

                    if cos_sim > max_sim:
                        max_sim = cos_sim
                        max_sim_id = i

        # print("max sim = ",max_sim)
        return self.branches[max_sim_id], max_sim_id

    def simple_learning_run(self, symbol_count, use_predefined_signal=True, verbose=False):

        windowSize = 4
        if use_predefined_signal:
            signal = [0, 1, 1, 0, 0, 0.25, 0.5, 0.75, 1, 0.75, 0.5, 0.25, 0, 1, 1, 0, 0, 0.25, 0.5, 0.75, 1, 0.75, 0.5,
                      0.25, 0,
                      0, 0, 0, 1, 0.5, 0, -0.5, -1, -0.5, 0, 0.5, 0, 0, 0, 0, 0, 1, 1, 0, 0, 1, 1, 0, 1, 0.5, 0, -0.5,
                      -1,
                      -0.5,
                      0, 0.5, 0, 0.5, 1, 0.5, 0, -0.5, -1, -0.5, 0, 1, 1, 0, 0, 0.5, 1, 0.5, 0, -0.5, -1, -0.5, 0, 0.5,
                      1,
                      0.5,
                      0, -0.5, -1, -0.5, 0, 0.5, 1, 0.5, 0, -0.5, -1, -0.5, 0, 0.5, 1, 0.5, 0, -0.5, -1, -0.5, 0, 0.5,
                      1,
                      0.5,
                      0, -0.5, -1, -0.5, 0, 1, 1, 0, 0, 1, 1, 0, 0, 0.5, 1, 0.5, 0, -0.5, -1, -0.5, 0, 0, 0, 0, 1, 0.5,
                      0,
                      -0.5,
                      -1, -0.5, 0, 0.5, 0, 0, 0, 0, 0, 1, 1, 0, 1, 0.5, 0, -0.5, -1, -0.5, 0, 0.5, 0, 0.5, 1, 0.5, 0,
                      -0.5,
                      -1,
                      -0.5, 0, 0, 0, 0, 0, 0.5, 1, 0.5, 0, -0.5, -1, -0.5, 0, 0.25, 0.5, 0.75, 1, 0.75, 0.5, 0.25, 0,
                      0.25,
                      0.5,
                      0.75, 1, 0.75, 0.5, 0.25, 0, 1, 1, 0, 0, 1, 1, 0, 0, 0.5, 1, 0.5, 0, -0.5, -1, -0.5, 0, 0.5, 1,
                      0.5,
                      0,
                      -0.5, -1, -0.5, 0, 0.25, 0.5, 0.75, 1, 0.75, 0.5, 0.25, 0, 1, 1, 0, 0, 0.25, 0.5, 0.75, 1, 0.75,
                      0.5,
                      0.25, 0, 0.5, 1, 0.5, 0, -0.5, -1, -0.5, 0, 0.5, 1, 0.5, 0, -0.5, -1, -0.5, 0, 1, 1, 0, 1, 0.5, 0,
                      -0.5,
                      -1, -0.5, 0, 0.5, 0, 0.5, 1, 0.5, 0, -0.5, -1, -0.5, 1, 0.5, 0, -0.5, -1, -0.5, 0, 0.5, 0, 1, 1,
                      0, 0,
                      0.5, 1, 0.5, 0, -0.5, -1, -0.5, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0.25, 0.5, 0.75, 1, 0.75, 0.5, 0.25, 0,
                      0,
                      0,
                      0, 1, 0.5, 0, -0.5, -1, -0.5, 0, 0.5, 0, 0.25, 0.5, 0.75, 1, 0.75, 0.5, 0.25, 0, 0.25, 0.5, 0.75,
                      1,
                      0.75,
                      0.5, 0.25, 0, 0.25, 0.5, 0.75, 1, 0.75, 0.5, 0.25, 0, 0.5, 1, 0.5, 0, -0.5, -1, -0.5, 0, 0.25,
                      0.5,
                      0.75,
                      1, 0.75, 0.5, 0.25, 0, 0, 0, 0, 0, 0.5, 1, 0.5, 0, -0.5, -1, -0.5, 0, 1, 1, 0, 0, 1, 1, 0, 0, 0,
                      0, 0,
                      0,
                      0, 0, 0, 0, 0.5, 1, 0.5, 0, -0.5, -1, -0.5, 0, 0, 0, 0, 0, 0.5, 1, 0.5, 0, -0.5, -1, -0.5, 0, 1,
                      1, 0,
                      1,
                      0.5, 0, -0.5, -1, -0.5, 0, 0.5, 0, 0.25, 0.5, 0.75, 1, 0.75, 0.5, 0.25, 1, 0.5, 0, -0.5, -1, -0.5,
                      0,
                      0.5,
                      0, 0.5, 1, 0.5, 0, -0.5, -1, -0.5, 1, 0.5, 0, -0.5, -1, -0.5, 0, 0.5, 0, 0.5, 1, 0.5, 0, -0.5, -1,
                      -0.5,
                      1, 0.5, 0, -0.5, -1, -0.5, 0, 0.5, 0, 1, 1, 0, 0, 0.25, 0.5, 0.75, 1, 0.75, 0.5, 0.25, 0, 0.25,
                      0.5,
                      0.75,
                      1, 0.75, 0.5, 0.25, 0, 0.5, 1, 0.5, 0, -0.5, -1, -0.5, 1, 0.5, 0, -0.5, -1, -0.5, 0, 0.5, 0, 0.25,
                      0.5,
                      0.75, 1, 0.75, 0.5, 0.25, 1, 0.5, 0, -0.5, -1, -0.5, 0, 0.5, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0.25, 0.5,
                      0.75,
                      1, 0.75, 0.5, 0.25, 0, 0.5, 1, 0.5, 0, -0.5, -1, -0.5, 0, 0.5, 1, 0.5, 0, -0.5, -1, -0.5, 0, 1, 1,
                      0,
                      0,
                      0.25, 0.5, 0.75, 1, 0.75, 0.5, 0.25, 0, 0.25, 0.5, 0.75, 1, 0.75, 0.5, 0.25, 0, 1, 1, 0, 0, 0.5,
                      1,
                      0.5,
                      0, -0.5, -1, -0.5, 0, 0.5, 1, 0.5, 0, -0.5, -1, -0.5, 0, 0.5, 1, 0.5, 0, -0.5, -1, -0.5, 0, 1, 1,
                      0,
                      0, 0,
                      0, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 0.25, 0.5, 0.75, 1, 0.75, 0.5, 0.25, 0, 0.5, 1, 0.5, 0, -0.5, -1,
                      -0.5,
                      0, 0.25, 0.5, 0.75, 1, 0.75, 0.5, 0.25, 0, 1, 1, 0, 1, 0.5, 0, -0.5, -1, -0.5, 0, 0.5, 1, 0.5, 0,
                      -0.5,
                      -1, -0.5, 0, 0.5, 0, 0.25, 0.5, 0.75, 1, 0.75, 0.5, 0.25, 1, 0.5, 0, -0.5, -1, -0.5, 0, 0.5, 0, 1,
                      1,
                      0,
                      1, 0.5, 0, -0.5, -1, -0.5, 0, 0.5, 0, 0.25, 0.5, 0.75, 1, 0.75, 0.5, 0.25, 0, 1, 1, 0, 0, 0.5, 1,
                      0.5,
                      0,
                      -0.5, -1, -0.5, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 1, 0.5, 0, -0.5, -1, -0.5, 0, 0.5, 0, 0.25,
                      0.5,
                      0.75,
                      1, 0.75, 0.5, 0.25, 0, 1, 1, 0, 1, 0.5, 0, -0.5, -1, -0.5, 0, 0.5, 0, 0, 0, 0, 1, 0.5, 0, -0.5,
                      -1,
                      -0.5,
                      0, 0.5, 0, 0.5, 1, 0.5, 0, -0.5, -1, -0.5, 0, 1, 1, 0, 0, 0.25, 0.5, 0.75, 1, 0.75, 0.5, 0.25, 0,
                      0,
                      0, 0,
                      0, 0.5, 1, 0.5, 0, -0.5, -1, -0.5, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0.25, 0.5, 0.75, 1, 0.75, 0.5, 0.25,
                      0,
                      1,
                      1, 0, 0, 1, 1, 0, 1, 0.5, 0, -0.5, -1, -0.5, 0, 0.5, 1, 0.5, 0, -0.5, -1, -0.5, 0, 0.5, 0, 0.25,
                      0.5,
                      0.75, 1, 0.75, 0.5, 0.25, 1, 0.5, 0, -0.5, -1, -0.5, 0, 0.5, 0, 0, 0, 0, 1, 0.5, 0, -0.5, -1,
                      -0.5, 0,
                      0.5, 0, 0, 0, 0, 0, 0.5, 1, 0.5, 0, -0.5, -1, -0.5, 0, 0.25, 0.5, 0.75, 1, 0.75, 0.5, 0.25, 1,
                      0.5, 0,
                      -0.5, -1, -0.5, 0, 0.5, 0, 0.25, 0.5, 0.75, 1, 0.75, 0.5, 0.25, 0, 1, 1, 0, 1, 0.5, 0, -0.5, -1,
                      -0.5,
                      0,
                      0.5, 0, 0.5, 1, 0.5, 0, -0.5, -1, -0.5, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.5, 1,
                      0.5, 0,
                      -0.5, -1, -0.5, 0, 0, 0, 0, 0, 0.25, 0.5, 0.75, 1, 0.75, 0.5, 0.25, 0, 0.5, 1, 0.5, 0, -0.5, -1,
                      -0.5,
                      0,
                      1, 1, 0, 0, 0.5, 1, 0.5, 0, -0.5, -1, -0.5, 0, 0.25, 0.5, 0.75, 1, 0.75, 0.5, 0.25, 0, 0.25, 0.5,
                      0.75, 1,
                      0.75, 0.5, 0.25, 0, 0.5, 1, 0.5, 0, -0.5, -1, -0.5, 0, 0.25, 0.5, 0.75, 1, 0.75, 0.5, 0.25, 0, 1,
                      1,
                      0, 1,
                      0.5, 0, -0.5, -1, -0.5, 0, 0.5, 1, 0.5, 0, -0.5, -1, -0.5, 0, 0.5, 0, 0.25, 0.5, 0.75, 1, 0.75,
                      0.5,
                      0.25,
                      0, 0.5, 1, 0.5, 0, -0.5, -1, -0.5, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0.25, 0.5, 0.75, 1, 0.75, 0.5, 0.25,
                      0,
                      0.5,
                      1, 0.5, 0, -0.5, -1, -0.5, 0, 0, 0, 0, 0, 0.5, 1, 0.5, 0, -0.5, -1, -0.5, 0, 0, 0, 0, 0, 0.25,
                      0.5,
                      0.75,
                      1, 0.75, 0.5, 0.25, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0.5, 1, 0.5, 0, -0.5, -1, -0.5, 0, 0, 0, 0, 0, 0.5,
                      1,
                      0.5,
                      0, -0.5, -1, -0.5, 0, 1, 1, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0.5, 1, 0.5, 0, -0.5, -1, -0.5, 1, 0.5,
                      0,
                      -0.5,
                      -1, -0.5, 0, 0.5, 0, 0.25, 0.5, 0.75, 1, 0.75, 0.5, 0.25, 0, 0, 0, 0, 1, 0.5, 0, -0.5, -1, -0.5,
                      0,
                      0.5,
                      0, 1, 1, 0, 0, 0.25, 0.5, 0.75, 1, 0.75, 0.5, 0.25, 0, 0.5, 1, 0.5, 0, -0.5, -1, -0.5, 0, 1, 1, 0,
                      0,
                      0.25, 0.5, 0.75, 1, 0.75, 0.5, 0.25, 1, 0.5, 0, -0.5, -1, -0.5, 0, 0.5, 1, 0.5, 0, -0.5, -1, -0.5,
                      0,
                      0.5,
                      0, 1, 1, 0, 0, 0.25, 0.5, 0.75, 1, 0.75, 0.5, 0.25, 1, 0.5, 0, -0.5, -1, -0.5, 0, 0.5, 0, 0.25,
                      0.5,
                      0.75,
                      1, 0.75, 0.5, 0.25, 0, 1, 1, 0, 1, 0.5, 0, -0.5, -1, -0.5, 0, 0.5, 0, 0, 0, 0, 1, 0.5, 0, -0.5,
                      -1,
                      -0.5,
                      0, 0.5, 0, 0.25, 0.5, 0.75, 1, 0.75, 0.5, 0.25, 0, 0.25, 0.5, 0.75, 1, 0.75, 0.5, 0.25, 0, 1, 1,
                      0, 1,
                      0.5, 0, -0.5, -1, -0.5, 0, 0.5]
        else:
            signal = self.veri.genSignalWithSimpleSymbols(symbol_count, add_nulls=True, verbose=verbose)

        print("train signal = ", signal, "\n")
        for i in range(len(signal) - windowSize + 1):
            temp = signal[i:i + windowSize]
            if self.use_wavelet:
                w = self.veri.getWaveletCoefs(input_data=temp)
            else:
                w = temp
            # print("data = ", temp, "w = ", w)
            self.learn(w)

    def simple_test_run(self, symbol_count, data = [],use_predefined_signal=True, learn = True, add_nulls=True, verbose=False):
        print("\ntest run ")
        windowSize = 4
        if use_predefined_signal:
            if len(data)>0:
                signal=data
            else:
                print("kare , kare, cos, cos, kare ")
                signal = [0, 1, 1, 0, 0, 1, 1, 0, 1, 0.5, 0, -0.5, -1, -0.5, 0, 0.5, 1, 0.5, 0, -0.5, -1, -0.5, 0, 0.5, 0,
                      1, 1, 0]

        else:
            signal = self.veri.genSignalWithSimpleSymbols(symbol_count, add_nulls=add_nulls, verbose=verbose)
        print("test signal = ", signal, "\n")
        print("Test data branche ids")
        output_branch_ids = []
        for i in range(len(signal) - windowSize + 1):
            temp = signal[i:i + windowSize]
            if self.use_wavelet:
                w = self.veri.getWaveletCoefs(input_data=temp)
            else:
                w = temp

            r, id = self.test(w)
            output_branch_ids.append(id)
            # print(id)
        if learn:
            self.learn_complex_symbols(output_branch_ids)
        else:
            self.learn_complex_symbols(output_branch_ids)

    def learn_complex_symbols(self, data):
        print("data: ", data)

        buffer = [0] * self.layer_count

        for i, d in enumerate(data):
            buffer.pop()
            buffer.insert(0, d)

            for i in range(self.layer_count):
                if buffer[i] == 0:
                    break
                if buffer[0:i + 1] in self.L[i]:
                    id = self.L[i].index(buffer[0:i + 1])
                    self.L_count[i][id] += 1
                else:
                    self.L[i].append(buffer[0:i + 1])
                    self.L_count[i].append(0)

    def find_complex_symbols(self, data):
        print("data: ", data)
        output = []
        for i in range(len(data)):
            output.append( [])
        buffer = [0] * self.layer_count

        for i, d in enumerate(data):
            buffer.pop()
            buffer.insert(0, d)

            for i in range(self.layer_count):
                if buffer[i] == 0:
                    break
                if buffer[0:i + 1] in self.L[i]:
                    id = self.L[i].index(buffer[0:i + 1])

                else:
                    self.L[i].append(buffer[0:i + 1])
                    self.L_count[i].append(0)

    def display_complex_relations(self):
        for i in range(self.layer_count):
            print("Layer", i)
            for j, d in enumerate(self.L[i]):
                print(j, self.L_count[i][j], d)
            print()

    def get_brach_status(self):
        print("Learnt branches")
        print("id freq branch")
        for i, b in enumerate(self.branches):
            print(i, self.branche_freq[i], b)


    def __init__(self):
        self.use_wavelet = False
        self.branches = []
        self.branche_freq = []
        self.start_index = 0
        self.veri = Veri()

        self.layer_count = 10
        self.L = []
        self.L_count = []
        for i in range(self.layer_count):
            self.L.append([])
            self.L_count.append([])

        # self.candidate_branches = []
        # self.candidate_branche_freq = []


def test1():
    symbols = Veri().predefinedSymbols()

    print(symbols)
    borders = [0]
    for key, value in symbols.items():
        plt.plot(value["val"], "-o")
        borders.append(borders[0] + value["p"])
        borders[0] = borders[-1]
    plt.legend(symbols)

    plt.show()


def test2():
    symbolCount = 20
    windowSize = 4
    # signal = veri.genSignalWithSimpleSymbols(symbolCount)
    signal = [0, 1, 1, 0, 0, 0.25, 0.5, 0.75, 1, 0.75, 0.5, 0.25, 0, 1, 1, 0, 0, 0.25, 0.5, 0.75, 1, 0.75, 0.5, 0.25, 0,
              0, 0, 0, 1, 0.5, 0, -0.5, -1, -0.5, 0, 0.5, 0, 0, 0, 0, 0, 1, 1, 0, 0, 1, 1, 0, 1, 0.5, 0, -0.5, -1, -0.5,
              0, 0.5, 0, 0.5, 1, 0.5, 0, -0.5, -1, -0.5, 0, 1, 1, 0, 0, 0.5, 1, 0.5, 0, -0.5, -1, -0.5, 0, 0.5, 1, 0.5,
              0, -0.5, -1, -0.5, 0, 0.5, 1, 0.5, 0, -0.5, -1, -0.5, 0, 0.5, 1, 0.5, 0, -0.5, -1, -0.5, 0, 0.5, 1, 0.5,
              0, -0.5, -1, -0.5, 0, 1, 1, 0, 0, 1, 1, 0, 0, 0.5, 1, 0.5, 0, -0.5, -1, -0.5, 0, 0, 0, 0, 1, 0.5, 0, -0.5,
              -1, -0.5, 0, 0.5, 0, 0, 0, 0, 0, 1, 1, 0, 1, 0.5, 0, -0.5, -1, -0.5, 0, 0.5, 0, 0.5, 1, 0.5, 0, -0.5, -1,
              -0.5, 0, 0, 0, 0, 0, 0.5, 1, 0.5, 0, -0.5, -1, -0.5, 0, 0.25, 0.5, 0.75, 1, 0.75, 0.5, 0.25, 0, 0.25, 0.5,
              0.75, 1, 0.75, 0.5, 0.25, 0, 1, 1, 0, 0, 1, 1, 0, 0, 0.5, 1, 0.5, 0, -0.5, -1, -0.5, 0, 0.5, 1, 0.5, 0,
              -0.5, -1, -0.5, 0, 0.25, 0.5, 0.75, 1, 0.75, 0.5, 0.25, 0, 1, 1, 0, 0, 0.25, 0.5, 0.75, 1, 0.75, 0.5,
              0.25, 0, 0.5, 1, 0.5, 0, -0.5, -1, -0.5, 0, 0.5, 1, 0.5, 0, -0.5, -1, -0.5, 0, 1, 1, 0, 1, 0.5, 0, -0.5,
              -1, -0.5, 0, 0.5, 0, 0.5, 1, 0.5, 0, -0.5, -1, -0.5, 1, 0.5, 0, -0.5, -1, -0.5, 0, 0.5, 0, 1, 1, 0, 0,
              0.5, 1, 0.5, 0, -0.5, -1, -0.5, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0.25, 0.5, 0.75, 1, 0.75, 0.5, 0.25, 0, 0, 0,
              0, 1, 0.5, 0, -0.5, -1, -0.5, 0, 0.5, 0, 0.25, 0.5, 0.75, 1, 0.75, 0.5, 0.25, 0, 0.25, 0.5, 0.75, 1, 0.75,
              0.5, 0.25, 0, 0.25, 0.5, 0.75, 1, 0.75, 0.5, 0.25, 0, 0.5, 1, 0.5, 0, -0.5, -1, -0.5, 0, 0.25, 0.5, 0.75,
              1, 0.75, 0.5, 0.25, 0, 0, 0, 0, 0, 0.5, 1, 0.5, 0, -0.5, -1, -0.5, 0, 1, 1, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0,
              0, 0, 0, 0, 0.5, 1, 0.5, 0, -0.5, -1, -0.5, 0, 0, 0, 0, 0, 0.5, 1, 0.5, 0, -0.5, -1, -0.5, 0, 1, 1, 0, 1,
              0.5, 0, -0.5, -1, -0.5, 0, 0.5, 0, 0.25, 0.5, 0.75, 1, 0.75, 0.5, 0.25, 1, 0.5, 0, -0.5, -1, -0.5, 0, 0.5,
              0, 0.5, 1, 0.5, 0, -0.5, -1, -0.5, 1, 0.5, 0, -0.5, -1, -0.5, 0, 0.5, 0, 0.5, 1, 0.5, 0, -0.5, -1, -0.5,
              1, 0.5, 0, -0.5, -1, -0.5, 0, 0.5, 0, 1, 1, 0, 0, 0.25, 0.5, 0.75, 1, 0.75, 0.5, 0.25, 0, 0.25, 0.5, 0.75,
              1, 0.75, 0.5, 0.25, 0, 0.5, 1, 0.5, 0, -0.5, -1, -0.5, 1, 0.5, 0, -0.5, -1, -0.5, 0, 0.5, 0, 0.25, 0.5,
              0.75, 1, 0.75, 0.5, 0.25, 1, 0.5, 0, -0.5, -1, -0.5, 0, 0.5, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0.25, 0.5, 0.75,
              1, 0.75, 0.5, 0.25, 0, 0.5, 1, 0.5, 0, -0.5, -1, -0.5, 0, 0.5, 1, 0.5, 0, -0.5, -1, -0.5, 0, 1, 1, 0, 0,
              0.25, 0.5, 0.75, 1, 0.75, 0.5, 0.25, 0, 0.25, 0.5, 0.75, 1, 0.75, 0.5, 0.25, 0, 1, 1, 0, 0, 0.5, 1, 0.5,
              0, -0.5, -1, -0.5, 0, 0.5, 1, 0.5, 0, -0.5, -1, -0.5, 0, 0.5, 1, 0.5, 0, -0.5, -1, -0.5, 0, 1, 1, 0, 0, 0,
              0, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 0.25, 0.5, 0.75, 1, 0.75, 0.5, 0.25, 0, 0.5, 1, 0.5, 0, -0.5, -1, -0.5,
              0, 0.25, 0.5, 0.75, 1, 0.75, 0.5, 0.25, 0, 1, 1, 0, 1, 0.5, 0, -0.5, -1, -0.5, 0, 0.5, 1, 0.5, 0, -0.5,
              -1, -0.5, 0, 0.5, 0, 0.25, 0.5, 0.75, 1, 0.75, 0.5, 0.25, 1, 0.5, 0, -0.5, -1, -0.5, 0, 0.5, 0, 1, 1, 0,
              1, 0.5, 0, -0.5, -1, -0.5, 0, 0.5, 0, 0.25, 0.5, 0.75, 1, 0.75, 0.5, 0.25, 0, 1, 1, 0, 0, 0.5, 1, 0.5, 0,
              -0.5, -1, -0.5, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 1, 0.5, 0, -0.5, -1, -0.5, 0, 0.5, 0, 0.25, 0.5, 0.75,
              1, 0.75, 0.5, 0.25, 0, 1, 1, 0, 1, 0.5, 0, -0.5, -1, -0.5, 0, 0.5, 0, 0, 0, 0, 1, 0.5, 0, -0.5, -1, -0.5,
              0, 0.5, 0, 0.5, 1, 0.5, 0, -0.5, -1, -0.5, 0, 1, 1, 0, 0, 0.25, 0.5, 0.75, 1, 0.75, 0.5, 0.25, 0, 0, 0, 0,
              0, 0.5, 1, 0.5, 0, -0.5, -1, -0.5, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0.25, 0.5, 0.75, 1, 0.75, 0.5, 0.25, 0, 1,
              1, 0, 0, 1, 1, 0, 1, 0.5, 0, -0.5, -1, -0.5, 0, 0.5, 1, 0.5, 0, -0.5, -1, -0.5, 0, 0.5, 0, 0.25, 0.5,
              0.75, 1, 0.75, 0.5, 0.25, 1, 0.5, 0, -0.5, -1, -0.5, 0, 0.5, 0, 0, 0, 0, 1, 0.5, 0, -0.5, -1, -0.5, 0,
              0.5, 0, 0, 0, 0, 0, 0.5, 1, 0.5, 0, -0.5, -1, -0.5, 0, 0.25, 0.5, 0.75, 1, 0.75, 0.5, 0.25, 1, 0.5, 0,
              -0.5, -1, -0.5, 0, 0.5, 0, 0.25, 0.5, 0.75, 1, 0.75, 0.5, 0.25, 0, 1, 1, 0, 1, 0.5, 0, -0.5, -1, -0.5, 0,
              0.5, 0, 0.5, 1, 0.5, 0, -0.5, -1, -0.5, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.5, 1, 0.5, 0,
              -0.5, -1, -0.5, 0, 0, 0, 0, 0, 0.25, 0.5, 0.75, 1, 0.75, 0.5, 0.25, 0, 0.5, 1, 0.5, 0, -0.5, -1, -0.5, 0,
              1, 1, 0, 0, 0.5, 1, 0.5, 0, -0.5, -1, -0.5, 0, 0.25, 0.5, 0.75, 1, 0.75, 0.5, 0.25, 0, 0.25, 0.5, 0.75, 1,
              0.75, 0.5, 0.25, 0, 0.5, 1, 0.5, 0, -0.5, -1, -0.5, 0, 0.25, 0.5, 0.75, 1, 0.75, 0.5, 0.25, 0, 1, 1, 0, 1,
              0.5, 0, -0.5, -1, -0.5, 0, 0.5, 1, 0.5, 0, -0.5, -1, -0.5, 0, 0.5, 0, 0.25, 0.5, 0.75, 1, 0.75, 0.5, 0.25,
              0, 0.5, 1, 0.5, 0, -0.5, -1, -0.5, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0.25, 0.5, 0.75, 1, 0.75, 0.5, 0.25, 0, 0.5,
              1, 0.5, 0, -0.5, -1, -0.5, 0, 0, 0, 0, 0, 0.5, 1, 0.5, 0, -0.5, -1, -0.5, 0, 0, 0, 0, 0, 0.25, 0.5, 0.75,
              1, 0.75, 0.5, 0.25, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0.5, 1, 0.5, 0, -0.5, -1, -0.5, 0, 0, 0, 0, 0, 0.5, 1, 0.5,
              0, -0.5, -1, -0.5, 0, 1, 1, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0.5, 1, 0.5, 0, -0.5, -1, -0.5, 1, 0.5, 0, -0.5,
              -1, -0.5, 0, 0.5, 0, 0.25, 0.5, 0.75, 1, 0.75, 0.5, 0.25, 0, 0, 0, 0, 1, 0.5, 0, -0.5, -1, -0.5, 0, 0.5,
              0, 1, 1, 0, 0, 0.25, 0.5, 0.75, 1, 0.75, 0.5, 0.25, 0, 0.5, 1, 0.5, 0, -0.5, -1, -0.5, 0, 1, 1, 0, 0,
              0.25, 0.5, 0.75, 1, 0.75, 0.5, 0.25, 1, 0.5, 0, -0.5, -1, -0.5, 0, 0.5, 1, 0.5, 0, -0.5, -1, -0.5, 0, 0.5,
              0, 1, 1, 0, 0, 0.25, 0.5, 0.75, 1, 0.75, 0.5, 0.25, 1, 0.5, 0, -0.5, -1, -0.5, 0, 0.5, 0, 0.25, 0.5, 0.75,
              1, 0.75, 0.5, 0.25, 0, 1, 1, 0, 1, 0.5, 0, -0.5, -1, -0.5, 0, 0.5, 0, 0, 0, 0, 1, 0.5, 0, -0.5, -1, -0.5,
              0, 0.5, 0, 0.25, 0.5, 0.75, 1, 0.75, 0.5, 0.25, 0, 0.25, 0.5, 0.75, 1, 0.75, 0.5, 0.25, 0, 1, 1, 0, 1,
              0.5, 0, -0.5, -1, -0.5, 0, 0.5]
    print("signal = ", signal)
    y = yapi()

    # print(len(signal))

    for i in range(len(signal) - windowSize + 1):
        temp = signal[i:i + windowSize]
        if y.use_wavelet:
            w = Veri().getWaveletCoefs(input_data=temp)
        else:
            w = temp
        # print("data = ", temp, "w = ", w)
        y.learn(w)
        # w_signal= w_signal + list(w)
        # print(i,temp , w)
        # plt.plot(w)
    # plt.show()

    for i, b in enumerate(y.branches):
        print(i, b, y.branche_freq[i])
    #     plt.plot(b)
    # plt.show()
    print("test ")
    r, id = y.test(w)
    print(w, r)
    # plt.plot(w)
    # plt.plot(r)
    # plt.show()

    symbolCount = 5
    signal = Veri().genSignalWithSimpleSymbols(symbolCount, verbose=True)
    print(signal)
    features = []
    for i in range(len(signal) - windowSize + 1):
        temp = signal[i:i + windowSize]
        if y.use_wavelet:
            w = Veri().getWaveletCoefs(input_data=temp)
        else:
            w = temp
        # print("data = ", temp, "w = ", w)
        # y.learn(w)
        r, id = y.test(w)
        print(id)

    # print("wsignal", w_signal)


def test3():
    y = yapi()
    y.getSymbols()

    y.simple_learning_run(symbol_count=100, use_predefined_signal=False, verbose=False)

    y.get_brach_status()
    y.simple_test_run(symbol_count=100, use_predefined_signal=False, add_nulls=True, verbose=False)
    y.simple_test_run(symbol_count=1000, use_predefined_signal=False, add_nulls=False, verbose=False)
    y.display_complex_relations()

    data =  [0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0]

    y.simple_test_run(data = data, symbol_count=100, use_predefined_signal=True, learn = False, verbose=True )

test3()
