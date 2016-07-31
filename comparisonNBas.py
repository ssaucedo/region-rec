#!/usr/bin/python3
import amfm_decompy.pYAAPT as pYAAPT
import amfm_decompy.basic_tools as basic
from modules import learning_characterization as LC
from modules.praat import scanAndGeneratePRAAT as gdata
from scipy.io import wavfile
import os
import matplotlib.pyplot as plt
import numpy as np
import sys
from subprocess import call

import warnings
warnings.filterwarnings('ignore')

class Region:
    def __init__(self, name,path_to_files):
        self.name = name
        self.path = path_to_files
        self.wav_files_id_list = []
        self.audio_char_list = []
    def add_audio_char(self,audio_c):
        self.audio_char_list.append(audio_c)
    def set_wav_files_id_list(self,RList):
        self.wav_files_id_list = RList

def retrieveCorporas(corpus_name,corpus_path, group,regions):
        file_dir = corpus_path + "/" + group + "/"
        original_dir = os.getcwd()
        os.chdir(file_dir)
        reg = Region(corpus_name,file_dir)
        for root, dirs, filenames in os.walk(os.getcwd()):
            reg.set_wav_files_id_list(filenames)
            regions.append(reg)
        os.chdir(original_dir)
        return regions

# YAAPT

def getPitch(file_dir, file_name):
    Fs, data = load_wav_file(file_dir, file_name)
    t = np.arange(0.0, len(data)/Fs, 1/Fs)
    signal = basic.SignalObj(data, Fs)
    pitch = pYAAPT.yaapt(signal)
    return pitch, t


# PRAAT

def getFormantMatrix(data_path):
    matrix = []
    for line in skipTwoReturnIterator(data_path):
        if line.strip():
            line = line.strip("\n").split("\t")
            matrix.append(list(map(int, line)))
    return matrix

def skipTwoReturnIterator(data_path):
    iterLines = iter(open(data_path))
    next(iterLines)
    next(iterLines)
    return iterLines

def getFormant(matrix, i):
    return [row[i] for row in matrix]


""" Script to generate descrptors;
files_path = ""
script_path = ""
gdata.extractData(files_path, script_path)
"""


""" Plot Pitch  """

ax1 = plt.subplot(211)
plt.title('Buenos Aires PITCH', loc='left')

data_path = "****/formant-log.txt"
pitchPRAAT = getFormant(getFormantMatrix(data_path), 0)
t = np.arange(0.0, len(pitchPRAAT), 1)
plt.plot(t, pitchPRAAT)


plt.subplot(212, sharex=ax1)
plt.title('Neuquen PITCH', loc='left')

data_path = "****/formant-log.txt"
pitchPRAAT = getFormant(getFormantMatrix(data_path), 0)
t = np.arange(0.0, len(pitchPRAAT), 1)
plt.plot(t, pitchPRAAT)



""" Plot Formant 1 """

ax1 = plt.subplot(211)
plt.title('Buenos Aires Formant 1', loc='left')

data_path =  "****/formant-log.txt"
pitchPRAAT = getFormant(getFormantMatrix(data_path), 1)
t = np.arange(0.0, len(pitchPRAAT), 1)
plt.plot(t, pitchPRAAT)


plt.subplot(212, sharex=ax1)
plt.title('Neuquen Formant 1', loc='left')

data_path = "****/formant-log.txt"
pitchPRAAT = getFormant(getFormantMatrix(data_path), 1)
t = np.arange(0.0, len(pitchPRAAT), 1)
plt.plot(t, pitchPRAAT)

plt.show()
