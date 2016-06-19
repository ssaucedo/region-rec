#!/usr/bin/python3
# -*- coding: utf-8 -*-

import numpy as np
from sklearn import preprocessing

from modules import learning_characterization as LC
from sklearn.mixture import GMM
import os


"""
Due to the lenght difference between the chinese and english spoken files
the analysis is limited to 140000 elements or 9 seconds.

Check LC.generate_MFCC()

English files average lenght = 14.70
Chinese files average lenght = 9.30

Linear Prediction Coding;

"""






class Region:
    def __init__(self, name,path_to_files):
        self.name = name
        self.path = path_to_files
        self.wav_files_id_list = []
        self.char_map = []
    def set_wav_files_id_list(self,RList):
        self.wav_files_id_list = RList


def retrieveCorporas(corpus_name,corpus_path,regions):
        original_dir = os.getcwd()
        os.chdir(corpus_path)
        reg = Region(corpus_name,corpus_path)
        for root, dirs, filenames in os.walk(os.getcwd()):
            reg.set_wav_files_id_list(filenames)
            regions.append(reg)
        os.chdir(original_dir)
        return regions




def concatenate_matrix(elements_keys, matrix, nroTest):
    concatenation = []
    for k in elements_keys[:nroTest]:
            if (len(concatenation) == 0):
                    concatenation = matrix[k]
            else:
                concatenation = np.concatenate((concatenation, matrix[k]))
    return concatenation

def main(l1,l2,nroTest,regions):
    g = 2048
    firstLanguaje = regions[l1]
    secondLanguaje = regions[l2]
    map_firstLanguaje = regions[l1].char_map
    map_secondLanguaje = regions[l2].char_map

    firstLanguajeConcatenation = concatenate_matrix(firstLanguaje.wav_files_id_list, map_firstLanguaje, nroTest)
    secondLanguajeConcatenation = concatenate_matrix(secondLanguaje.wav_files_id_list, map_secondLanguaje, nroTest)

    gmmFirstLanguaje = GMM(n_components = g , covariance_type = 'diag')
    gmmFirstLanguaje.fit(firstLanguajeConcatenation)
    gmmSecondLanguaje = GMM(n_components = g , covariance_type = 'diag')
    gmmSecondLanguaje.fit(secondLanguajeConcatenation)

    totalFIR = 0
    totalSEC = 0


    for k in firstLanguaje.wav_files_id_list[nroTest:]:
            vot_firstLanguaje = 0
            vot_secondLanguaje = 0
            firstLanguajeMFCC = map_firstLanguaje[k]
            firstLanguajeMFCC = preprocessing.scale(firstLanguajeMFCC)
            prediccion_firstLanguaje = gmmFirstLanguaje.predict_proba(firstLanguajeMFCC)
            prediccion_secondLanguaje = gmmSecondLanguaje.predict_proba(firstLanguajeMFCC)
            weight_firstLanguaje = gmmFirstLanguaje.weights_.reshape(1, -1).T
            weight_secondLanguaje = gmmSecondLanguaje.weights_.reshape(1, -1).T
            prob_firstLanguaje = np.dot(prediccion_firstLanguaje, weight_firstLanguaje)
            prob_secondLanguaje = np.dot(prediccion_secondLanguaje, weight_secondLanguaje)
            for i in range(0,prob_firstLanguaje.shape[0]):
                if(prob_firstLanguaje[i][0] > prob_secondLanguaje[i][0]):
                    vot_firstLanguaje += 1
                else:
                    vot_secondLanguaje += 1
            if(vot_firstLanguaje > vot_secondLanguaje):
                totalFIR +=1
            else:
                totalSEC += 1
    print(firstLanguaje.name +str(totalFIR))
    print(secondLanguaje.name +str(totalSEC))



chinese_dir = "/home/santiagosau/proyectos/PFI/repository/chinese_corpus/THCHS30_2015/wavFiles/"
english_dir = "/home/santiagosau/proyectos/PFI/repository/english_corpus/LibriSpeech/wavFiles/"

regions = []
regions = retrieveCorporas("Chinese",chinese_dir,regions)
regions = retrieveCorporas("English",english_dir,regions)

pr = 0.1
#regions = LC.generate_MFFC_for_regions(regions,pr)  #---------> generate and save MFFC for each .wav of each region. Returns the regions with the char_map field updated
regions = LC.get_LPC_for_regions(regions)
