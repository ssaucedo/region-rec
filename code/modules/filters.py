## Libreria filtros.py
## Funciones para realizar los filtrados necesarios de las seniales

# Importo librerias sistema
import numpy as np
import scipy.signal as scipy

# Importo librerias ajenas
# Bajada de https://github.com/jameslyons/python_speech_features
from code.modules import sigproc

# Importo libreria propias
from code.modules import espectro

# Parametros de la ventana (globales)
vent_largo = 256 # L, Largo ventana
vent_avance = 85 # M, Avance entre ventanas

# Parametros del filtrado pre-enfasis
coefPreEnfasis = 0.98

# Parametros del filtro FIR
frec_corte_superior = 3450 # Hz (voz humana hasta 3,4 KHz)  -->  Se usa este valor para que el filtro en 3400 Hz tenga ganancia aprox 1
frec_corte_inferior = 250 # Hz (voz humana desde 300 Hz) ------> Se usa este valor para que el filtro en 300 Hz tenga ganancia aprox 1
ancho_transicion = 100 # Hz
att_filtro = 60 # dB

# Funcion que aplica un filtro de pre enfasis al audio. El coeficiente esta definido como variable global al comienzo del archivo
def filtroPreEnfasis( dat ):
    dat_filtro = sigproc.preemphasis( dat, coefPreEnfasis)
    return dat_filtro
          
 
# Funcion para aplicar filtrado tipo FIR, utilizando ventana de Kaiserord                     
def filtroFIR( dat, Fs ):
    # Referencia de aplicacion sacada de:
    # http://scipy-cookbook.readthedocs.org/items/FIRFilter.html
    
    N, beta, taps, nyq_rate = coeficientesFiltroFIR( Fs ) # No uso todos los parametros devueltos aqui, pero la funcion la hice generica

    # Use lfilter to filter x with the FIR filter.
    # Documentacion funcion: http://docs.scipy.org/doc/scipy-0.14.0/reference/generated/scipy.signal.lfilter.html#scipy.signal.lfilter
    # lfilter(b, a, x, axis=-1, zi=None)
    datFiltrado = scipy.lfilter(taps, 1.0, dat)
    
    return datFiltrado

# Funcion que calcula los coeficientes del filtro a utilizar (Kaiserord, uno de los mejores)   
def coeficientesFiltroFIR( Fs ):
    # The Nyquist rate of the signal.
    nyq_rate = Fs / 2.0

    # Compute the order and Kaiser parameter for the FIR filter.
    # Documentacion funcion: http://docs.scipy.org/doc/scipy-0.14.0/reference/generated/scipy.signal.kaiserord.html#scipy.signal.kaiserord
    # kaiserord(ripple, width)
    N, beta = scipy.kaiserord(att_filtro, (ancho_transicion/nyq_rate) )

    # Use firwin with a Kaiser window to create a lowpass FIR filter.
    # Documentacion funcion: http://docs.scipy.org/doc/scipy-0.14.0/reference/generated/scipy.signal.firwin.html
    # firwin(numtaps, cutoff, width=None, window='hamming', pass_zero=True, scale=True, nyq=1.0)
    taps = scipy.firwin(N, (frec_corte_inferior/nyq_rate, frec_corte_superior/nyq_rate), window=('kaiser', beta), pass_zero=False )
    
    return N, beta, taps, nyq_rate

    
    
    