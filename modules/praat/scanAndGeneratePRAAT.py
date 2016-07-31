#!/usr/bin/python3
import numpy as np
from scipy.io import wavfile
import os
from subprocess import call
import shutil

def retrieveCorporas(files_path):
        original_dir = os.getcwd()
        os.chdir(files_path)
        reg = []
        for root, dirs, filenames in os.walk(os.getcwd()):
            reg = filenames
        os.chdir(original_dir)
        return reg


files_path = ""
files_names = retrieveCorporas(files_path)
script_path = ""
analysisPath = files_path + "filesAnalysis/"
files_analysis_path = analysisPath +  "name" + "/"



os.makedirs(analysisPath)


for name in files_names:
        file_path = files_path + name
        files_analysis_path = analysisPath +  name + "/"
        os.makedirs(files_analysis_path)
        call(["praat",  script_path , file_path , files_analysis_path])
