import matplotlib.pyplot as plt
import numpy as np


def old():
    np.random.seed(19680801)
    data = np.random.random((50, 5, 5))

    fig, ax = plt.subplots()

    for i in range(len(data)):
        ax.cla()
        #data[i]=data[i]>0.5

        ax.imshow(data[i],cmap='RdBu')
        ax.set_title("frame {}".format(i))
        # Note that using time.sleep does *not* work here!
        plt.pause(0.1)

        #plt.colorbar(cmap='RdBu')
    print ("done")

import os
import sys
sys.path.append(os.path.abspath(os.path.dirname(__file__) + '/' + '../modules'))
sys.path.append(os.path.abspath(os.getcwd() + '/test1/test1/modules'))


from BaseStructure import BaseStructure
from Components import Abstract, Context1, Context2, Actuator, Spektron
from Veri import Veri
from scipy.io import wavfile
import matplotlib.pyplot as plt
import numpy as np
import playsound
import pyAudioAnalysis
#from python_speech_features import mfcc, logfbank
from fnmatch import fnmatch
import os
from python_speech_features import mfcc, logfbank



def browseFolder( root, pattern):
    print(root)
    lista = []
    # pattern = "*.lyr"
    for path, subdirs, files in os.walk(root):
        # print( path
        for name in files:
            if fnmatch(name.lower(), pattern):
                lista.append(os.path.join(path, name))
    return lista

def learned_weights_x():
    ans = []
    with open('weights.txt', 'r') as weight_file:
        lines = weight_file.readlines()
        for i in lines[0].split('\t'):
            ans.append(float(i))
    return ans

agac = BaseStructure()

x=(browseFolder("../dune","*.txt"))
for i,f in enumerate(x):
    print(i,f, f.split("\\")[-1])
    with open(f, 'r',errors='ignore') as metin:
        lines = metin.readlines()
        print("number of sentences = ",len(lines))
    print(lines[:10])
    for j in range(10):
        print(lines[j].lower().split(),)
        kelimeler = lines[j].lower().split()
        if(len(kelimeler)>0):
            for k,harfler  in enumerate(kelimeler):
                agac.addBranch(list(harfler))

agac.plotGraph()

    #break



