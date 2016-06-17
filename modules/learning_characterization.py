# -*- coding: utf-8 -*-
import numpy as np
import warnings
import scipy.signal as scipy
import scipy.fftpack as scyfft
import os

from sklearn.externals.six.moves import xrange
from sklearn.mixture import GMM

import modules.base
import modules.espectro
import modules.cargaAudio as ca
import modules.filters as fl

# Ignora DeprecationWarning
warnings.filterwarnings("ignore", category=DeprecationWarning)



# Carpetas de archivos MFCC y GMM


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

powerRelation = 0.2 #------> Relacion potencia promedio - potencia ventana. Se procesan aquellas ventanas con potencia mayor al [powerRelation * Potencia-promedio]


def save_info( data, name, wav_id):


    os.chdir("D:\\UADE\PFI_workspace\\TestMultilingual")
    filename = 'caracterizaciones/' + name + '/MFCC_' + wav_id + '/'
    os.makedirs(filename)

    np.savetxt( (filename + 'MFCC_' + wav_id + '.csv' ), data, fmt='%.18e', delimiter=';', newline='\n', header=('REGION: ' + name + 'PROCESSED IDs:'   +wav_id), footer='', comments='# ')
    return

# Levanta MFCC de archivo CSV
def load_info(name, wav_id):
    os.chdir("D:\\UADE\\PFI_workspace\\TestMultilingual\\")
    filename = 'caracterizaciones/' + name + '/MFCC_' + wav_id + '/'
    mfcc = np.loadtxt( (filename + 'MFCC_' + wav_id + '.csv' ), delimiter=';',  skiprows=1, ndmin=0) # Evito la primera row por contener el comentario
    return mfcc


def generate_GMM_region_list( regions, mode ):
    for region in regions:
        if(mode == "directo"):
           calculoGMM( region.mfcc_concatenation, region.name, mode)
    return



def generate_MFFC_region_list( regions, pr):
    mapList = []
    for region in  regions :
        mapMFCC = {}
        mfcc = []
        for file_id in region.wav_files_id_list:
                mfcc  = generate_MFCC(region.name , file_id, pr)
                save_info( mfcc, region.name, file_id)
                mapMFCC[file_id] = mfcc
        mapList.append(mapMFCC)
    return  mapList


def get_MFFC_region_map_list( regions):
    mapList = []
    for region in  regions :
        mapMFCC = {}
        mfcc = []
        for file_id in region.wav_files_id_list:
                mfcc = load_info( region.name, file_id)
                mapMFCC[file_id] = mfcc
        mapList.append(mapMFCC)
    return  mapList



def generate_MFCC(name , file_id, pr):
     Fs , dat = ca.load_wav_file(name , file_id)
     dat = fl.filtroPreEnfasis(dat)
     dat = fl.filtroFIR(dat, Fs)
     print(name + str(len(dat)))
     if(len(dat) > 4000):
         dat = dat[:4000]
     # CalculateMFCC
     WDFFT, Fs = getFilteredWindowedDFFT(dat, Fs, pr)
     MF = getMF(WDFFT , nroFiltrosMEL, Fs)
     MFCC = getMFCC(MF)

     return MFCC



"""
FUNCIONES  MFFC
"""
def getFilteredWindowedDFFT(data , Fs, pr):
     N = len(data);
     WDFFT = []
     i = 0
     idx = 0
     P = 0
     wn = scipy.get_window(ventana,L)
     while idx < N-L-1:
          y = data[idx:idx+L] * wn
          P = P + espectro.powerOf(y,L)
          idx = idx + M
          i = i +1
     P = P/i  #-------->    Potencia por ventana *promedio
     idx = 0
     while idx < N-L-1:
          y = data[idx:idx+L] * wn
          Pv =espectro.powerOf(y,L)
          if (Pv > P* pr):
              f, Y = espectro.spectrum(y,Fs)
              WDFFT.append(Y)
          idx = idx + M
     return np.asarray(WDFFT),Fs


def getMF(WDFFT , nroFiltrosMEL ,Fs):
     filterParam = (WDFFT[0].shape[0]-1)*2
     fb = base.get_filterbanks(nroFiltrosMEL,filterParam,Fs)
     MFArray = np.zeros((WDFFT.shape[0], nroFiltrosMEL))
     fl = np.asarray(list(range(0, WDFFT[0].shape[0])))
     #for window in WDFFT:
     for i in xrange(0,WDFFT.shape[0]):
         for j in xrange(0,nroFiltrosMEL):
              fl1= fb[j,fl]
              MF = fl1 * WDFFT[i]   #---------> MF = Ventana * Filtro de Mel
              MFp =  espectro.powerOf(MF,filterParam/2) #---------> MFp = energia de MF
              np.put(MFArray[i],[j],MFp) #---------> Agrego la potencia MFp en la posicion que corresponde.
     return MFArray


def getMFCC(MF):
     cm = np.log10(MF)  #Cepstrum
     MFCC = []
     for c in cm:
         mffcCoef  = scyfft.idct(c)   #Coseno inverso
         MFCC.append(mffcCoef)
     MFCC = np.array(MFCC)
     return MFCC
