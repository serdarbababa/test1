
import scipy.io.wavfile as wavfile
import numpy as np
import pyaudio
import wave
import matplotlib.pyplot as plt
from scipy.signal import butter, lfilter
from scipy.signal import freqz
from fnmatch import fnmatch
import os
import random

from scipy.io import wavfile
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile
from python_speech_features import mfcc, logfbank


class Utilities:
    def browseFolder(self,root,pattern):
        print( root)
        lista = []
        #pattern = "*.lyr"
        for path, subdirs, files in os.walk(root):
            #print( path
            for name in files:
                if fnmatch(name.lower(), pattern):
                    lista.append( os.path.join(path, name))
        return lista

    def testSound(self,filename):




        # define stream chunk
        chunk = 1024

        # open a wav format music
        f = wave.open(filename, "rb")
        # instantiate PyAudio
        p = pyaudio.PyAudio()
        # open stream
        stream = p.open(format=p.get_format_from_width(f.getsampwidth()),
                        channels=f.getnchannels(),
                        rate=f.getframerate(),
                        output=True)
        # read data
        data = f.readframes(chunk)
        tempData = data

        # play stream
        while data:
            stream.write(data)
            data = f.readframes(chunk)
            #tempData.append(data)
        print( tempData[0:10])

            # stop stream
        stream.stop_stream()
        stream.close()

        # close PyAudio
        p.terminate()


        # fs = 10000
        # p = pyaudio.PyAudio()
        # volume = 0.6
        # stream = p.open(format=pyaudio.paInt16,
        #             channels=1,
        #             rate=fs*2,
        #             output=True)
        #
        # # # play. May repeat with different volume values (if done interactively)
        # stream.write(volume * data[0:fs*10])
        #
        # stream.stop_stream()
        # stream.close()
        # p.terminate()
    def butter_bandpass(self,lowcut, highcut, fs, order=5):
        ##### these two functions are used to filter the data
        nyq = 0.5 * fs
        low = lowcut / nyq
        high = highcut / nyq
        b, a = butter(order, [low, high], btype='band')
        return b, a

    def butter_bandpass_filter(self,data, lowcut, highcut, fs, order=5):
        b, a = self.butter_bandpass(lowcut, highcut, fs, order=order)
        #print( b,a
        y = lfilter(b, a, data)
        res = []

        return np.asanyarray( y, np.float)

    def testButter(self):
        import numpy as np

        from scipy.signal import freqz

        # Sample rate and desired cutoff frequencies (in Hz).
        fs = 5000.0
        lowcut = 500.0
        highcut = 1250.0

        # Plot the frequency response for a few different orders.
        plt.figure(1)
        plt.clf()
        for order in [3, 6, 9]:
            b, a = self.butter_bandpass(lowcut, highcut, fs, order=order)
            w, h = freqz(b, a, worN=2000)
            plt.plot((fs * 0.5 / np.pi) * w, abs(h), label="order = %d" % order)

        plt.plot([0, 0.5 * fs], [np.sqrt(0.5), np.sqrt(0.5)],
                 '--', label='sqrt(0.5)')
        plt.xlabel('Frequency (Hz)')
        plt.ylabel('Gain')
        plt.grid(True)
        plt.legend(loc='best')

        # Filter a noisy signal.
        T = 0.05
        nsamples = T * fs
        t = np.linspace(0, T, nsamples, endpoint=False)
        a = 0.02
        f0 = 600.0
        x = 0.1 * np.sin(2 * np.pi * 1.2 * np.sqrt(t))
        x += 0.01 * np.cos(2 * np.pi * 312 * t + 0.1)
        x += a * np.cos(2 * np.pi * f0 * t + .11)
        x += 0.03 * np.cos(2 * np.pi * 2000 * t)
        plt.figure(2)
        plt.clf()
        plt.plot(t, x, label='Noisy signal')

        y = self.butter_bandpass_filter(x, lowcut, highcut, fs, order=6)
        plt.plot(t, y, label='Filtered signal (%g Hz)' % f0)
        plt.xlabel('time (seconds)')
        plt.hlines([-a, a], 0, T, linestyles='--')
        plt.grid(True)
        plt.axis('tight')
        plt.legend(loc='upper left')

        plt.show()

    ## read wave file and return data and rate
    def readSound(self,filename):
        fs, data = wavfile.read(filename)
        #print( data[215600:215700]
        print( fs, len(data))
        return [fs, data]

    # read wave file and return data from given chanel and rate
    def readSoundChanel(self,filename, chanel):
        fs, data = wavfile.read(filename)
        res = []

        if type(data[0]) is  np.ndarray:
            for dat in data:
                res.append(float(dat[chanel])/32767)
        else:
            res = float(data)

        print( fs, len(data))
        return [fs, np.asarray(res,dtype=np.float)]

    # write wav file from given data and rate
    def writeSound(self,filename, rate, data):
        wavfile.write(filename, rate, data)

    # plays a wav file
    def playSound(self,filename):
        # define stream chunk
        chunk = 1024

        # open a wav format music
        f = wave.open(filename, "rb")
        # instantiate PyAudio
        p = pyaudio.PyAudio()
        # open stream
        stream = p.open(format=p.get_format_from_width(f.getsampwidth()),
                        channels=f.getnchannels(),
                        rate=f.getframerate(),
                        output=True)
        # read data
        data = f.readframes(chunk)

        # play stream
        while data:
            stream.write(data)
            data = f.readframes(chunk)

            # stop stream
        stream.stop_stream()
        stream.close()

        # close PyAudio
        p.terminate()

    # bandpass filter
    def filterData(self,veri, rate, lowcut, highcut):
        #lowcut = 1000
        #highcut = 1200
        y = self.butter_bandpass_filter(veri, lowcut, highcut, rate, order=2)
        return y

    # plot data
    def plotData(self,data):
        plt.plot(data)
        plt.show()

    # plot multiple data sets
    def plotMultiData(self,data):
        for dat in data:
            plt.plot(dat)
        plt.show()

    # plots
    def seeFilterResponse(self,lowcut, highcut, rate):
        plt.figure(1)
        plt.clf()
        for order in [3, 6, 9]:
            b, a = self.butter_bandpass(lowcut, highcut, rate, order=order)
            w, h = freqz(b, a, worN=2000)
            plt.plot((rate * 0.5 / np.pi) * w, abs(h), label="order = %d" % order)

        plt.plot([0, 0.5 * rate], [np.sqrt(0.5), np.sqrt(0.5)],'--', label='sqrt(0.5)')
        plt.show()

    def playData(self,data, rate):
        # instantiate PyAudio (1)
        p = pyaudio.PyAudio()

        # open stream (2), 2 is size in bytes of int16
        stream = p.open(format=p.get_format_from_width(2),
                        channels=1,
                        rate=rate,
                        output=True)

        # play stream (3), blocking call
        #stream.write(data)
        chunk = 1024
        index=0
        data1 = data[index*chunk:(index+1)*chunk-1]
        # play stream

        for i in range(len(data)/chunk-1):
            stream.write(data1.tostring())
            index = index+1
            data1 = data[index*chunk:(index+1)*chunk-1]

        # stop stream (4)
        stream.stop_stream()
        stream.close()

        # close PyAudio (5)
        p.terminate()

    # save data to file
    def writeDataToFile(self,data, filename):
        file = open(filename, "w")

        for dat in data:
            file.write(str(dat)+"\n")
        file.close()

    def writeWavDataToFile(self,data,rate, filename):
        file = open(filename, "w")
        file.write(str(rate) + "\n")
        for dat in data:
            file.write(str(dat)+"\n")
        file.close()
    # read data from file
    def readDataFromFile(self,filename):
        data=[]
        file = open(filename, "r")
        for line in file:
            line = line.replace("\n","")
            #print( line
            data.append(float(line))
        return data
        #return np.asanyarray(data)


    def readWavDataFromFile(self,filename):
        data = []
        rate = 0
        file = open(filename, "r")
        count = 0
        for line in file:
            line = line.replace("\n", "")
            # print( line
            if count == 0:
                rate = int(line)
            else:
                data.append(float(line))
            count = 1
        return [rate, data]
        # return np.asanyarray(data)


    def functionsTest(self):
        filename = '/Users/ser/OneDrive/teza/datasets/test_mono_8000Hz_16bit_PCM.wav'
        filename = '/Users/ser/OneDrive/spectron/data/brian.wav'

        [rate, veri] = self.readSoundChanel(filename, 0)
        outFilename = '/Users/ser/OneDrive/teza/datasets/test2.wav'
        # playSound(outFilename)
        # writeSound('/Users/ser/OneDrive/teza/datasets/test.wav', rate, veri)
        y = self.filterData(veri, rate, 1000, 10000)
        self.playData(y, rate)
        self.plotMultiData([veri, y])

        filename = '/Users/ser/OneDrive/teza/datasets/test.txt'

        self.writeDataToFile(range(100), filename)

        data = self.readDataFromFile(filename)
        print( data)


    def prepareDataForNet(self,root,voice, fileCount):
        #root = '/Users/ser/OneDrive/spectron/data'
        #root = '/Users/ser/Desktop/calismalar/teza/data'
        #root = '/Users/ser/OneDrive/spectron/data'



        rate = 0
        data = 0

        for i in range(fileCount):
            filename = root + voice + ".wav_" + str(i) + ".txt"
            print( filename)
            [rate, z] = self.readWavDataFromFile(filename )
            print( i, len(z))
            if i==0:
                data = np.zeros(len(z))
                i=1
            data = data+np.asanyarray(z,dtype=np.float)

        self.writeSound('/Users/ser/Desktop/calismalar/teza/outputxx.wav',rate,data)



    def prepareData(self):
        root = '/Users/ser/OneDrive/spectron/data'
        files = self.browseFolder(root, '*.wav')

        print( "found files")
        for f in files:
            print( f)
            [rate, veri] = self.readSoundChanel(f, 0)
            bandwidth=500
            startFreq1=20
            startFreq2 = 320
            startFreq3 = 1070
            startFreq4 = 5070
            bandCount = 75
            for i in range(bandCount):
                print( i)
                y=0
                if i <15:
                    bandwidth=20
                    y = self.filterData(veri, rate, startFreq1+ i * bandwidth, startFreq1+ (i + 1) * bandwidth)

                elif i < 30:
                    bandwidth = 50
                    y = self.filterData(veri, rate, startFreq2 + (i-15) * bandwidth, startFreq2 + (i -15+ 1) * bandwidth)

                elif i<50   :
                    bandwidth=200
                    y = self.filterData(veri, rate, startFreq3 + (i - 30) * bandwidth, startFreq3 + (i - 30 + 1) * bandwidth)

                else   :
                    bandwidth=500
                    y = self.filterData(veri, rate, startFreq4 + (i - 50) * bandwidth, startFreq4 + (i - 50+ 1) * bandwidth)
            self.writeWavDataToFile(y, rate, f+"_"+str(i)+".txt")
            break

    def prepareData1(self,root, target):

        files = self.browseFolder(root, '*.wav')

        print( "found files",len(files))

        for f in files[1:]:
            print( f)
            [rate, veri] = self.readSoundChanel(f, 0)

            bandwidth = 50
            startFreq1=100
            bandCount =10
            for i in range(bandCount):
                print( i, rate, startFreq1 + i * bandwidth, startFreq1 + (i + 1) * bandwidth)
                y = self.filterData(veri, rate, startFreq1 + i * bandwidth, startFreq1 + (i + 1) * bandwidth)
                filename = target + f.split('/')[-1]+"_"+str(i)+".txt"
                print( filename)
                self.writeWavDataToFile(y, rate, filename)


    def feedNet(self, root,  band, voice):

        f= root+voice+'_'+str(band)+'.txt'
        [rate, z] = self.readWavDataFromFile(f )
        return z

u = Utilities()
#f = u.browseFolder(r"C:\serdar\data\train\train\audio\digits", "*.wav")
f= [r"C:\serdar\data\train\train\audio\digits\four\747e69fd_nohash_0.wav",
    r"C:\serdar\data\train\train\audio\digits\five\afd53389_nohash_0.wav",
    r"C:\serdar\data\train\train\audio\digits\eight\df6bd83f_nohash_1.wav",
    r"C:\serdar\data\train\train\audio\digits\one\08ab8082_nohash_0.wav",
    r"C:\serdar\data\train\train\audio\digits\eight\f17be97f_nohash_2.wav",
    r"C:\serdar\data\train\train\audio\digits\four\2f666bb2_nohash_1.wav",
    r"C:\serdar\data\train\train\audio\digits\six\3b852f6f_nohash_0.wav",
    r"C:\serdar\data\train\train\audio\digits\seven\fd395b74_nohash_3.wav",
    r"C:\serdar\data\train\train\audio\digits\two\172dc2b0_nohash_1.wav",
    r"C:\serdar\data\train\train\audio\digits\zero\0fa1e7a9_nohash_0.wav"]
print(len(f))
for i in range(2):
    k=i #random.randint(0,len(f))
    print(f[k])
    u.playSound(f[k])
    [rate, z] = u.readSound(f[k])
    #normalizing # audio_signal = audio_signal / np.power(2, 15)

    plt.plot(z)
    plt.show()
    #########################

    frequency_sampling, audio_signal = wavfile.read(f[k])
    features_mfcc = mfcc(audio_signal, frequency_sampling,numcep=8)

    print('\nMFCC:\nNumber of windows =', features_mfcc.shape[0])
    print('Length of each feature =', features_mfcc.shape[1])

    features_mfcc = features_mfcc.T
    cax = plt.matshow(features_mfcc)
    plt.title('MFCC')
    plt.colorbar(cax)
    plt.show()
    # for i in range(len(features_mfcc)):
    #     plt.plot(features_mfcc[i,:])
    #     plt.title("MFCC "+str(i))
    #     plt.show()


    # filterbank_features = logfbank(audio_signal, frequency_sampling)
    #
    # print('\nFilter bank:\nNumber of windows =', filterbank_features.shape[0])
    # print('Length of each feature =', filterbank_features.shape[1])
    #
    # filterbank_features = filterbank_features.T
    # plt.matshow(filterbank_features)
    # plt.title('Filter bank')
    # plt.show()
    #





