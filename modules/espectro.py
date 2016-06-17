# -*- coding: utf-8 -*-
import numpy as np
from scipy import fft as scyfft

# Funcion para calcular el espectro de la señal
def spectrum( y , fs ):
    L, NFFT = calculoNFFTyL( y )
    
    Y = scyfft(y, NFFT) / L
    f = fs/2*np.linspace(0,1,NFFT/2+1)
    P = abs(Y[0:NFFT/2+1])

    return f,P

# Funcion para calcular el espectro de la señal, pero en dB
def spectrum_dB( y, fs ):
    L, NFFT = calculoNFFTyL( y )
    
    Y = scyfft(y, NFFT) / L
    f = fs/2*np.linspace(0,1,NFFT/2+1)
    P = 20*np.log10( abs(Y[0:NFFT/2+1]) )

    return f,P

# Funcion auxiliar para calculo de los parametros de la transformada    
def calculoNFFTyL( dat ):    
    dat = dat.flatten(1)
    #L = 4*len(dat) CUAL CONVIENE??
    L = len(dat)
    NFFT = 2**np.ceil( np.log2(L) )
    NFFT = np.int(NFFT)
    
    return L, NFFT

# Sub-funcion auxiliar para tener solamente el retorno del parametro NFFT
def calculoNFFT( dat ):
    L, NFFT = calculoNFFTyL(dat)
    
    return NFFT

# Sub-funcion auxiliar para tener solamente el retorno del parametro L
def calculoL( dat ):
    L, NFFT = calculoNFFTyL(dat)
    
    return L

# Funcion para calcular la potencia de la señal
def powerOf(array,NFFT):
    p =  1.0/NFFT * np.sum(np.square(array))
    
    return p