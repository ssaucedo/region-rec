## Libreria cargaAudio.py
## Funciones para cargar audios a memoria normalizandolos en el proceso

# Importo librerias del sistema
from scipy.io import wavfile
import numpy as np
import sys
import os

global_audio_folder = '/corpora'

def load_wav_file( region_name , file_id_number ):
    os.chdir("D:\\UADE\\PFI_workspace\\TestMultilingual\\corpora\\")
    Fs, dat = wavfile.read(region_name+'/256K/'+ file_id_number)
    normDat = normalize_audio( dat ).astype(float)
    return Fs, normDat

def normalize_audio( dat ):
    maxDat = np.max( np.abs(dat) )
    datNorm = dat.astype(float)/np.float(maxDat)
    return datNorm
