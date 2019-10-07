import re

import matplotlib.pyplot as plt
# %matplotlib inline
import IPython.display as ipd  # To play sound in the notebook
import scipy.io.wavfile
from scipy.fftpack import dct

# importing all the dependencies
import pandas as pd  # data frame
import numpy as np  # matrix math
from glob import glob  # file handling
import librosa  # audio manipulation
from sklearn.utils import shuffle  # shuffling of data
import os  # interation with the OS
from random import sample  # random selection
from tqdm import tqdm

# fixed param
PATH = '../input/train/audio/'
PATH = '/Users/serdar/Documents/data/train/audio'


def load_files(path):
    # write the complete file loading function here, this will return
    # a dataframe having files and labels
    # loading the files
    train_labels = os.listdir(PATH)
    train_labels.remove('_background_noise_')
    train_labels.remove('.DS_Store')

    labels_to_keep = ['yes', 'no', 'up', 'down', 'left', 'right', 'on', 'off', 'stop', 'go', 'silence']
    # labels_to_keep = ['bed', 'bird', 'cat', 'dog', 'down', 'eight', 'five', 'four', 'go','happy', 'house', 'left', 'marvin', 'nine', 'no', 'off', 'on', 'one', 'right', 'seven', 'sheila', 'six', 'stop', 'three', 'tree',       'two', 'up', 'wow', 'yes', 'zero']
    train_file_labels = dict()
    for label in train_labels:
        files = os.listdir(PATH + '/' + label)
        for f in files:
            train_file_labels[label + '/' + f] = label

    train = pd.DataFrame.from_dict(train_file_labels, orient='index')
    train = train.reset_index(drop=False)
    train = train.rename(columns={'index': 'file', 0: 'folder'})
    train = train[['folder', 'file']]
    train = train.sort_values('file')
    train = train.reset_index(drop=True)

    def remove_label_from_file(label, fname):
        return path + label + '/' + fname[len(label) + 1:]

    train['file'] = train.apply(lambda x: remove_label_from_file(*x), axis=1)
    train['label'] = train['folder'].apply(lambda x: x if x in labels_to_keep else 'unknown')

    labels_to_keep.append('unknown')

    return train, labels_to_keep


# Writing functions to extract the data, script from kdnuggets:
# www.kdnuggets.com/2016/09/urban-sound-classification-neural-networks-tensorflow.html
def extract_feature(path):
    X, sample_rate = librosa.load(path)
    stft = np.abs(librosa.stft(X))
    mfccs = np.mean(librosa.feature.mfcc(y=X, sr=sample_rate, n_mfcc=40).T, axis=0)
    chroma = np.mean(librosa.feature.chroma_stft(S=stft, sr=sample_rate).T, axis=0)
    mel = np.mean(librosa.feature.melspectrogram(X, sr=sample_rate).T, axis=0)
    contrast = np.mean(librosa.feature.spectral_contrast(S=stft, sr=sample_rate).T, axis=0)
    tonnetz = np.mean(librosa.feature.tonnetz(y=librosa.effects.harmonic(X), sr=sample_rate).T, axis=0)
    return mfccs, chroma, mel, contrast, tonnetz


def parse_audio_files(files, word2id, unk=False):
    # n: number of classes
    features = np.empty((0, 193))
    one_hot = np.zeros(shape=(len(files), word2id[max(word2id)]))
    print(one_hot.shape)
    for i in tqdm(range(len(files))):
        f = files[i]
        mfccs, chroma, mel, contrast, tonnetz = extract_feature(f)
        ext_features = np.hstack([mfccs, chroma, mel, contrast, tonnetz])
        features = np.vstack([features, ext_features])
        if unk == True:
            l = word2id['unknown']
            one_hot[i][l] = 1.
        else:
            l = word2id[f.split('/')[-2]]
            one_hot[i][l] = 1.
    return np.array(features), one_hot


# we now convert it to spertogram
# goto: https://www.kaggle.com/davids1992/data-visualization-and-investigation
def log_specgram(audio, sample_rate, window_size=10, step_size=10, eps=1e-10):
    nperseg = int(round(window_size * sample_rate / 1e3))
    noverlap = int(round(step_size * sample_rate / 1e3))
    # print(nperseg)
    # print(noverlap)
    a, b, spec = scipy.signal.spectrogram(audio,
                                          fs=sample_rate,
                                          window='hann',
                                          nperseg=nperseg,
                                          noverlap=noverlap,
                                          detrend=False)
    x = np.log(spec.T.astype(np.float32) + eps)
    y = np.zeros([spec.shape[1], 10])
    z = np.zeros([spec.shape[1], 10])
    # print(x.shape)
    # print(y.shape)
    y[:, 0] = x[:, 0]

    for i in range(1, 10):
        y[:, i] = sum(abs(x[:, (i - 1) * 9 + 1:(i) * 9].T))
        z[:, i] = y[:, i] > 40

    return x, y, z




train, labels_to_keep = load_files(PATH)

# making word2id dict
word2id = dict((c, i) for i, c in enumerate(sorted(labels_to_keep)))

# get some files which will be labeled as unknown
unk_files = train.loc[train['label'] == 'unknown']['file'].values
unk_files = sample(list(unk_files), 1000)

print(train["folder"].unique())

print(word2id)

print(unk_files[:10])

print(train.sample(5))

files = train.loc[train['label'] != 'unknown']['file'].values
print(len(files))
print(files[:10])

# playing around with the data for now
train_audio_path = PATH
filename = '/tree/24ed94ab_nohash_0.wav'  # --> 'Yes'
# filename = '/tree/1a073312_nohash_0.wav'
sample_rate, audio = scipy.io.wavfile.read(str(train_audio_path) + filename)

plt.figure(figsize=(15, 4))
plt.plot(audio)
plt.show()
ipd.Audio(audio, rate=sample_rate)

windowsNr= 50
print(audio.shape[0])
step = int(audio.shape[0]/windowsNr)

y=np.zeros(audio.shape)
z=np.zeros(windowsNr)
for i in range(windowsNr):
    temp = audio[i*step : (i+1)*step]
    z[i]=np.std(temp)
    #if(z[i]>500):
    y[i*step : (i+1)*step] = (temp  - np.mean(temp) )/np.std(temp)
    #else:
    #    y[i*step : (i+1)*step] = (temp*1.0)
plt.plot(z)
plt.show()



spectrogram,bands,thr= log_specgram(audio, sample_rate, 10, 3)
spec = spectrogram.T
print(spec.shape)
plt.figure(figsize = (15,4))
plt.imshow(spec, aspect='auto', origin='lower')
#print(bands.shape)
plt.figure(figsize = (15,4))
plt.imshow(bands.T, aspect='auto', origin='lower')
plt.colorbar()

plt.figure(figsize = (15,4))
plt.imshow(thr.T, aspect='auto', origin='lower')
plt.colorbar()

plt.show()


spectrogram,bands,thr= log_specgram(audio, sample_rate, 10, 3)
spec = spectrogram.T
print(spec.shape)
plt.figure(figsize = (15,4))
plt.imshow(spec, aspect='auto', origin='lower')
#print(bands.shape)
plt.figure(figsize = (15,4))
plt.imshow(bands.T, aspect='auto', origin='lower')
plt.colorbar()

plt.figure(figsize = (15,4))
plt.imshow(thr.T, aspect='auto', origin='lower')
plt.colorbar()


plt.show()