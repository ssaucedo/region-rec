#!/usr/bin/python3
import amfm_decompy.pYAAPT as pYAAPT
import amfm_decompy.basic_tools as basic
from modules import learning_characterization as LC
from scipy.io import wavfile
import os
import matplotlib.pyplot as plt
import numpy as np
import sys

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

def load_wav_file( file_path , file_name ):
    Fs, dat = wavfile.read(file_path + file_name)
    return Fs, dat


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

def getPitch(file_dir, file_name):
    Fs, data = load_wav_file(file_dir, file_name)
    t = np.arange(0.0, len(data)/Fs, 1/Fs)
    signal = basic.SignalObj(data, Fs)
    pitch = pYAAPT.yaapt(signal)
    return pitch, t


"""
buenos_aires_path = "***/Buenos Aires"
neuquen_path = "***/Neuquen"
group = "Parte_1"

regions = []
regions = retrieveCorporas("buenos aires",buenos_aires_path, group, regions)
regions = retrieveCorporas("neuquen",neuquen_path, group, regions)


buenos_aires_group_path = regions[0].path
neuquen_group_path = regions[1].path

buenos_aires_ids = regions[0].wav_files_id_list
neuquen_ids = regions[1].wav_files_id_list

plt.subplot(211)
pitch , t = getPitch(buenos_aires_group_path, buenos_aires_ids[2] )
plt.plot(t, pitch.values)


plt.subplot(212)
pitch , t = getPitch(neuquen_group_path, neuquen_ids[2] )
plt.plot(t, pitch.values)
plt.show


for file_name in regions[n].wav_files_id_list:
         print(file_name)
         Fs, dat = load_wav_file(regions[n].path, file_name)
         p = p + len(dat)
print(p/(len(regions[n].wav_files_id_list)))
"""


Fs, data = wavfile.read((os.getcwd()+"/resources/english_speech.wav"))
data = LC.normalize_audio(data[1:len(data)-2])


t = np.arange(0.0, len(data)/Fs, 1/Fs)
signal = basic.SignalObj(data, Fs)
pitch = pYAAPT.yaapt(signal)

plt.plot(t, pitch.values)
plt.show()
