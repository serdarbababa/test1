import sys
sys.path.append(r"C:\Users\dellCTA\OneDrive\teza\code\simplu\test1\modules")

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

def calistirma():

    spek = Spektron()

    for i in range(1000):
        spek.oneBeat(verbose=False)

    spek.displaySpektron()
    operations = [spek.v.symbols[1], spek.v.symbols[10], spek.v.symbols[1], spek.v.symbols[14]]
    print(operations)
    for i in range(len(operations)):
        print(i, operations[i])


    for symbol in operations:
        spek.checkOperation1(symbol)

    for symbol in operations:
        spek.checkOperation1(symbol)


    for symbol in operations:
        spek.checkOperation1(symbol)
    # operations = spek.getInstantOperationInput()
    # [print(i, operations[i]) for i in range(len(operations))]
    # # for count, item in enumerate(operations):
    # #    spek.oneBeat(symbol=item, verbose=True)
    # for symbol in operations:
    #     spek.oneComplexBeat(symbol)
    # for symbol in operations[:-1]:
    #     spek.checkOperation(symbol)
    #
    # for i in range(100):
    #     operations = spek.getInstantOperationInput()
    #     for symbol in operations:
    #         spek.oneComplexBeat(symbol)

    #spek.displaySpektron()

def waveletDonustur(data):
    v = Veri()
    m=[]
    plt.plot(data)
    plt.show()
    for i in range(int(len(data)/128)):
        coefs = v.getWaveletCoefs(data[i*128:(i+1)*128])
        m.append(coefs)
    print(len(m))
    m=np.array(m).T

    cax= plt.matshow(m)
    plt.xlabel("Time")
    plt.ylabel("Wavelet Coeff")
    plt.colorbar(cax)
    plt.title('wavelet')
    plt.show()

    plt.hist(m)
    plt.show()


def deney():
    f = [r"C:\serdar\data\train\train\audio\digits\four\747e69fd_nohash_0.wav",
         r"C:\serdar\data\train\train\audio\digits\five\afd53389_nohash_0.wav",
         r"C:\serdar\data\train\train\audio\digits\eight\df6bd83f_nohash_1.wav",
         r"C:\serdar\data\train\train\audio\digits\one\08ab8082_nohash_0.wav",
         r"C:\serdar\data\train\train\audio\digits\eight\f17be97f_nohash_2.wav",
         r"C:\serdar\data\train\train\audio\digits\four\2f666bb2_nohash_1.wav",
         r"C:\serdar\data\train\train\audio\digits\six\3b852f6f_nohash_0.wav",
         r"C:\serdar\data\train\train\audio\digits\seven\fd395b74_nohash_3.wav",
         r"C:\serdar\data\train\train\audio\digits\two\172dc2b0_nohash_1.wav",
         r"C:\serdar\data\train\train\audio\digits\zero\0fa1e7a9_nohash_0.wav"]
    f = browseFolder(r"C:\serdar\data\train\train\audio\digits", "*.wav")
    abstract = Abstract(16, 0)
    leafs = []
    for i in range(10):
        playsound.playsound(f[i])
        print(i,f[i])

        frequency_sampling, data = wavfile.read(f[i])
        plt.plot(data)
        plt.show()
        continue


        features_mfcc = mfcc(data, frequency_sampling, numcep=16,)
        a=features_mfcc[0]
        for i in range(int(len(features_mfcc) )):
            abstract_output = abstract.learnSymbol(features_mfcc[i ], False)
            leafs.append(abstract_output)


        #print('\nMFCC:\nNumber of windows =', features_mfcc.shape[0])
        #print('Length of each feature =', features_mfcc.shape[1])

        #features_mfcc = features_mfcc.T
        #cax = plt.matshow(features_mfcc)
        #plt.title('MFCC')
        #plt.colorbar(cax)
        #plt.show()

        print("son yaprak", leafs[-1])
        continue

        #playsound.playsound(f[i])
        print(i,f[i])
        #waveletDonustur(data)

        for i in range(int(len(data) / 128)):
            abstract_output = abstract.learnSymbol(data[i * 128:(i + 1) * 128], False)
            leafs.append(abstract_output)
        print("son yaprak" , leafs[-1])
    plt.plot(leafs)
    plt.show()
        #features_mfcc = mfcc(audio_signal, frequency_sampling, numcep=8)
        #print('\nMFCC:\nNumber of windows =', features_mfcc.shape[0])
        #print('Length of each feature =', features_mfcc.shape[1])


        #


def deney1():
    f = [r"C:\serdar\data\train\train\audio\digits\four\747e69fd_nohash_0.wav",
         r"C:\serdar\data\train\train\audio\digits\five\afd53389_nohash_0.wav",
         r"C:\serdar\data\train\train\audio\digits\eight\df6bd83f_nohash_1.wav",
         r"C:\serdar\data\train\train\audio\digits\one\08ab8082_nohash_0.wav",
         r"C:\serdar\data\train\train\audio\digits\eight\f17be97f_nohash_2.wav",
         r"C:\serdar\data\train\train\audio\digits\four\2f666bb2_nohash_1.wav",
         r"C:\serdar\data\train\train\audio\digits\six\3b852f6f_nohash_0.wav",
         r"C:\serdar\data\train\train\audio\digits\seven\fd395b74_nohash_3.wav",
         r"C:\serdar\data\train\train\audio\digits\two\172dc2b0_nohash_1.wav",
         r"C:\serdar\data\train\train\audio\digits\zero\0fa1e7a9_nohash_0.wav"]
    #f = browseFolder(r"C:\serdar\data\train\train\audio\digits", "*.wav")
    abstract = Abstract(128, 0)
    leafs = []
    for i in range(2):
        #playsound.playsound(f[i])
        frequency_sampling, data = wavfile.read(f[i])



        #playsound.playsound(f[i])
        print(i,f[i])
        #waveletDonustur(data)

        for i in range(int(len(data) / 128)):
            abstract_output = abstract.learnSymbol(data[i * 128:(i + 1) * 128], False)
            leafs.append(abstract_output)
        print("son yaprak" , leafs[-1])
    plt.plot(leafs)
    plt.show()
        #features_mfcc = mfcc(audio_signal, frequency_sampling, numcep=8)
        #print('\nMFCC:\nNumber of windows =', features_mfcc.shape[0])
        #print('Length of each feature =', features_mfcc.shape[1])


        #

def features():
    f = [r"C:\serdar\data\train\train\audio\digits\four\747e69fd_nohash_0.wav",
         r"C:\serdar\data\train\train\audio\digits\five\afd53389_nohash_0.wav",
         r"C:\serdar\data\train\train\audio\digits\eight\df6bd83f_nohash_1.wav",
         r"C:\serdar\data\train\train\audio\digits\one\08ab8082_nohash_0.wav",
         r"C:\serdar\data\train\train\audio\digits\eight\f17be97f_nohash_2.wav",
         r"C:\serdar\data\train\train\audio\digits\four\2f666bb2_nohash_1.wav",
         r"C:\serdar\data\train\train\audio\digits\six\3b852f6f_nohash_0.wav",
         r"C:\serdar\data\train\train\audio\digits\seven\fd395b74_nohash_3.wav",
         r"C:\serdar\data\train\train\audio\digits\two\172dc2b0_nohash_1.wav",
         r"C:\serdar\data\train\train\audio\digits\zero\0fa1e7a9_nohash_0.wav"]
    frequency_sampling, audio_signal = wavfile.read(f[1])
    features_mfcc = mfcc(audio_signal, frequency_sampling, numcep=16)

    print('\nMFCC:\nNumber of windows =', features_mfcc.shape[0])
    print('Length of each feature =', features_mfcc.shape[1])

    features_mfcc = features_mfcc.T
    cax = plt.matshow(features_mfcc)
    plt.title('MFCC')
    plt.colorbar(cax)
    plt.show()

calistirma()
#features()
#deney()


#v = Veri()
#symbol = v.genInstantSymbol(True)
#abstract = Abstract(128,0)
#abstract_output = abstract.learnSymbol(symbol,True)



