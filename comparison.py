#!/usr/bin/python3
import amfm_decompy.pYAAPT as pYAAPT
import amfm_decompy.basic_tools as basic
from modules import learning_characterization as LC
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



script_path = "***.script"
name =   "english_speech.wav"
"""
files_path = "***/resources/"
analysisPath = files_path + "filesAnalysis/"
os.makedirs(analysisPath)
file_path = files_path + name
files_analysis_path = analysisPath + name + "/"
os.makedirs(files_analysis_path)
call(["praat",  script_path , file_path , files_analysis_path])
"""

Fs, data = wavfile.read((os.getcwd()+"/resources/english_speech.wav"))


ax1 = plt.subplot(211)
data_path = "***/formant-log.txt"
matrix = getFormantMatrix(data_path)
pitchPRAAT = getFormant(matrix, 0)
t = np.arange(0.0, 1200, 1)
t = np.arange(0.0, len(data)/Fs, (len(data)/Fs)/1200)

plt.plot(t, pitchPRAAT[:1200])


""" PRINT YAAPT """
plt.subplot(212, sharex=ax1)
data = LC.normalize_audio(data[1:len(data)-2])
t = np.arange(0.0, len(data)/Fs, 1/Fs)
signal = basic.SignalObj(data, Fs)
pitch = pYAAPT.yaapt(signal)
plt.plot(t, pitch.values)
plt.show()
