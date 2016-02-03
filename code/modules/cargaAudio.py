## Libreria cargaAudio.py
## Funciones para cargar audios a memoria normalizandolos en el proceso

# Importo librerias del sistema
from scipy.io import wavfile
import numpy as np
import sys

global_audio_folder = 'wav_files_grouped_by_region'

def load_wav_file( region_name , file_id_number ):
    Fs, dat = wavfile.read(global_audio_folder+'/'+region_name+'/reg-'+ file_id_number +'.wav')
    normDat = normalize_audio( dat ).astype(float)
    return Fs, normDat    
        
def normalize_audio( dat ):
    maxDat = np.max( np.abs(dat) )
    datNorm = dat.astype(float)/np.float(maxDat)
    return datNorm
    