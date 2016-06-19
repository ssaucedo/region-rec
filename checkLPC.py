#!/usr/bin/python3
import amfm_decompy.pYAAPT as pYAAPT
import amfm_decompy.basic_tools as basic
from modules import learning_characterization as LC
from scipy.io import wavfile
import os



"""
The scope of this .py is check the best ALGORITHM for pitch detection

Options
     - Propietary
     - pYAAPT      --->  http://bjbschmitt.github.io/AMFM_decompy/pYAAPT.html


   Usage pYAAPT:
        signal = basic.SignalObj('path_to_sample.wav')
        pitch = pYAAPT.yaapt(signal)

"""


Fs , dat = wavfile.read(os.getcwd() + "english_speech.wav")
dat = LC.normalize_audio(dat)
signal = basic.SignalObj('path_to_sample.wav')
