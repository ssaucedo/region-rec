#!/usr/bin/python3
# -*- coding: utf-8 -*-

import numpy as np
import warnings
import scipy.signal as scipy
from scipy.io import wavfile
import os

import modules.base as base
import modules.espectro as espectro
import modules.filters as fl

import amfm_decompy.pYAAPT as pYAAPT
import amfm_decompy.basic_tools as basic


# Ignora DeprecationWarning
warnings.filterwarnings("ignore", category=DeprecationWarning)


# Constantes para LPC
idx = 0
SW2d = []
T = []
p = 12 # orden del filtro lineal

# Constantes para MFCC
ventana = "hamming"
L = 360
M = 120
nroFiltrosMEL = 13



def save_info(data, name, wav_id):
    filename = 'characterizations/' + name + '/MFCC_' + wav_id + '/'
    os.makedirs(filename)
    np.savetxt( (filename + 'MFCC_' + wav_id + '.csv' ), data, fmt='%.18e', delimiter=';', newline='\n', header=('REGION: ' + name + 'PROCESSED IDs:'   +wav_id), footer='', comments='# ')
    return

# Levanta MFCC de archivo CSV
def load_info(name, wav_id):
    filename = 'characterizations/' + name + '/MFCC_' + wav_id + '/'
    mfcc = np.loadtxt( (filename + 'MFCC_' + wav_id + '.csv' ), delimiter=';',  skiprows=1, ndmin=0) # Evito la primera row por contener el comentario
    return mfcc


def generate_GMM_region_list( regions, mode ):
    for region in regions:
        if(mode == "directo"):
           calculoGMM( region.mfcc_concatenation, region.name, mode)
    return



def generate_LPC_for_regions( regions, pr):
    for region in  regions :
        mapLPC = {}
        lpc = []
        for file_id in region.wav_files_id_list:
                lpc  = generate_LPC(region.path, file_id, pr)
                save_info( lpc, region.name, file_id)
                mapLPC[file_id] = lpc
        region.char_map = mapLPC
    return  regions


def get_LPC_for_regions( regions):
    for region in  regions :
        mapLPC = {}
        lpc = []
        for file_id in region.wav_files_id_list:
                lpc = load_info(region.name, file_id)
                mapLPC[file_id] = lpc
        region.char_map = mapLPC
    return  regions



def generate_LPC(file_path, file_id, pr):
     """ Limited dat because of differences with Chinese and English corporas"""
     Fs , dat = wavfile.read(file_path + file_id)
     if(len(dat) > 140000):
         dat = dat[:140000]
     dat = normalize_audio(dat).astype(float)
     signal = basic.SignalObj('path_to_sample.wav')
     pitch = pYAAPT.yaapt(signal)


     #MFCC = mfcc(dat)
     return LPC


def normalize_audio(dat):
    maxDat = np.max(np.abs(dat))
    datNorm = dat.astype(float)/np.float(maxDat)
    return datNorm
