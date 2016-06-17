## Libreria filtros.py
## Funciones para realizar los filtrados necesarios de las seniales

# Importo librerias sistema
import numpy as numpy
import scipy.signal as scipy

# Importo librerias ajenas
# Bajada de https://github.com/jameslyons/python_speech_features
import modules.sigproc

# Importo libreria propias
import modules.espectro
import math

def framesig(sig,frame_len,frame_step,winfunc=lambda x:numpy.ones((x,))):
    """Frame a signal into overlapping frames.

    :param sig: the audio signal to frame.
    :param frame_len: length of each frame measured in samples.
    :param frame_step: number of samples after the start of the previous frame that the next frame should begin.
    :param winfunc: the analysis window to apply to each frame. By default no window is applied.
    :returns: an array of frames. Size is NUMFRAMES by frame_len.
    """
    slen = len(sig)
    frame_len = int(round(frame_len))
    frame_step = int(round(frame_step))
    if slen <= frame_len:
        numframes = 1
    else:
        numframes = 1 + int(math.ceil((1.0*slen - frame_len)/frame_step))

    padlen = int((numframes-1)*frame_step + frame_len)

    zeros = numpy.zeros((padlen - slen,))
    padsignal = numpy.concatenate((sig,zeros))

    indices = numpy.tile(numpy.arange(0,frame_len),(numframes,1)) + numpy.tile(numpy.arange(0,numframes*frame_step,frame_step),(frame_len,1)).T
    indices = numpy.array(indices,dtype=numpy.int32)
    frames = padsignal[indices]
    win = numpy.tile(winfunc(frame_len),(numframes,1))
    return frames*win


def deframesig(frames,siglen,frame_len,frame_step,winfunc=lambda x:numpy.ones((x,))):
    """Does overlap-add procedure to undo the action of framesig.

    :param frames: the array of frames.
    :param siglen: the length of the desired signal, use 0 if unknown. Output will be truncated to siglen samples.
    :param frame_len: length of each frame measured in samples.
    :param frame_step: number of samples after the start of the previous frame that the next frame should begin.
    :param winfunc: the analysis window to apply to each frame. By default no window is applied.
    :returns: a 1-D signal.
    """
    frame_len = round(frame_len)
    frame_step = round(frame_step)
    numframes = numpy.shape(frames)[0]
    assert numpy.shape(frames)[1] == frame_len, '"frames" matrix is wrong size, 2nd dim is not equal to frame_len'

    indices = numpy.tile(numpy.arange(0,frame_len),(numframes,1)) + numpy.tile(numpy.arange(0,numframes*frame_step,frame_step),(frame_len,1)).T
    indices = numpy.array(indices,dtype=numpy.int32)
    padlen = (numframes-1)*frame_step + frame_len

    if siglen <= 0: siglen = padlen

    rec_signal = numpy.zeros((padlen,))
    window_correction = numpy.zeros((padlen,))
    win = winfunc(frame_len)

    for i in range(0,numframes):
        window_correction[indices[i,:]] = window_correction[indices[i,:]] + win + 1e-15 #add a little bit so it is never zero
        rec_signal[indices[i,:]] = rec_signal[indices[i,:]] + frames[i,:]

    rec_signal = rec_signal/window_correction
    return rec_signal[0:siglen]

def magspec(frames,NFFT):
    """Compute the magnitude spectrum of each frame in frames. If frames is an NxD matrix, output will be NxNFFT.

    :param frames: the array of frames. Each row is a frame.
    :param NFFT: the FFT length to use. If NFFT > frame_len, the frames are zero-padded.
    :returns: If frames is an NxD matrix, output will be NxNFFT. Each row will be the magnitude spectrum of the corresponding frame.
    """
    complex_spec = numpy.fft.rfft(frames,NFFT)
    return numpy.absolute(complex_spec)

def powspec(frames,NFFT):
    """Compute the power spectrum of each frame in frames. If frames is an NxD matrix, output will be NxNFFT.

    :param frames: the array of frames. Each row is a frame.
    :param NFFT: the FFT length to use. If NFFT > frame_len, the frames are zero-padded.
    :returns: If frames is an NxD matrix, output will be NxNFFT. Each row will be the power spectrum of the corresponding frame.
    """
    return 1.0/NFFT * numpy.square(magspec(frames,NFFT))

def logpowspec(frames,NFFT,norm=1):
    """Compute the log power spectrum of each frame in frames. If frames is an NxD matrix, output will be NxNFFT.

    :param frames: the array of frames. Each row is a frame.
    :param NFFT: the FFT length to use. If NFFT > frame_len, the frames are zero-padded.
    :param norm: If norm=1, the log power spectrum is normalised so that the max value (across all frames) is 1.
    :returns: If frames is an NxD matrix, output will be NxNFFT. Each row will be the log power spectrum of the corresponding frame.
    """
    ps = powspec(frames,NFFT);
    ps[ps<=1e-30] = 1e-30
    lps = 10*numpy.log10(ps)
    if norm:
        return lps - numpy.max(lps)
    else:
        return lps

def preemphasis(signal,coeff=0.95):
    """perform preemphasis on the input signal.

    :param signal: The signal to filter.
    :param coeff: The preemphasis coefficient. 0 is no filter, default is 0.95.
    :returns: the filtered signal.
    """
    return numpy.append(signal[0],signal[1:]-coeff*signal[:-1])



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
    dat_filtro = preemphasis( dat, coefPreEnfasis)
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
