# -*- coding: utf-8 -*-
import numpy as np

class Region:
    def __init__(self, name):
        self.name = name
        self.wav_files_id_list = []
        self.mfcc_concatenation = []
        self.lpc_concatenation = []
    def add_audio_id(self,id):
        self.wav_files_id_list.append(id)    
    def set_wav_files_id_list(self, list):
        self.wav_files_id_list = list    
    def set_mfcc_concatenation(self, mfcc_con):
        self.mfcc_concatenation = mfcc_con
    def set_lpc_concatenation(self, lpc_con):
        self.lpc_concatenation = lpc_con

def get_regions_from_csv(csv_directory):
    csv_file = np.loadtxt(csv_directory, dtype = 'string', delimiter=';', ndmin=0)
    regions = []
    for row in csv_file:
          reg = Region(row[0])
          np.delete(row, 0)
          reg.set_wav_files_id_list(filter(None,(np.delete(row, 0))))           
          regions.append(reg)
    return regions
