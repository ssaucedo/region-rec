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

"""



def save_info( data, name, wav_id):

    print("SAVING: "+name)
    os.chdir("D://UADE//PFI_workspace//TestMultilingual")
    filename = 'caracterizaciones/' + name + '/MFCC_' + wav_id + '/'
    os.makedirs(filename)

    np.savetxt( (filename + 'MFCC_' + wav_id + '.csv' ), data, fmt='%.18e', delimiter=';', newline='\n', header=('REGION: ' + name + 'PROCESSED IDs:'   +wav_id), footer='', comments='# ')
    return


class Region:
    def __init__(self, name,path_to_files):
        self.name = name
        self.path = path_to_files
        self.wav_files_id_list = []
        self.char_map = []
    def set_wav_files_id_list(self,RList):
        self.wav_files_id_list = RList


class audio_char:
    def __init__(self, wav_id,MFCC):
        self.id = wav_id
        self.wav_files_id_list = MFCC

def get_regions_from_csv(csv_directory):
    csv_file = np.loadtxt(csv_directory, dtype = 'string', delimiter=';', ndmin=0)
    regions = []
    for row in csv_file:
          reg = Region(row[0])
          np.delete(row, 0)
          reg.set_wav_files_id_list(filter(None,(np.delete(row, 0))))
          regions.append(reg)
    return regions



def retrieveCorporas(corpus_name,corpus_path,regions):
        original_dir = os.getcwd()
        os.chdir(corpus_path)
        reg = Region(corpus_name,corpus_path)
        for root, dirs, filenames in os.walk(os.getcwd()):
            reg.set_wav_files_id_list(filenames)
            regions.append(reg)
        os.chdir(original_dir)
        return regions


def saveGMM( gmm, region_name, wav_id, OAA, N ):

    converged_arr = np.array( [gmm.converged_] )
    print("SAVING GMM")

    filename = 'caracterizaciones/' + region_name + '/MFCC_' + wav_id + '/GMM_'+ str(N)+ '/'
    os.makedirs(filename)

    np.savetxt( (filename + 'weights' + '.csv'), gmm.weights_, fmt='%.18e', delimiter=';', newline='\n', header=('GMM_weights from ' + str(OAA)  + "  must be compared with"+ wav_id ))
    np.savetxt( (filename + 'means' + '.csv'), gmm.means_, fmt='%.18e', delimiter=';', newline='\n', header=('GMM_means from ' + region_name + wav_id + "must be compared with"+ str(OAA) ))
    np.savetxt( (filename + 'covars' + '.csv'), gmm.covars_, fmt='%.18e', delimiter=';', newline='\n', header=('GMM_covars from ' + region_name + wav_id + "must be compared with"+ str(OAA)))
    np.savetxt( (filename + 'converged' + '.csv'), converged_arr , fmt='%.18e', delimiter=';', newline='\n', header=('GMM_converged from ' + region_name + wav_id + "must be compared with"+ str(OAA)))

    return



def saveMainGMM( gmm, region_name , N):

    converged_arr = np.array( [gmm.converged_] )
    print("SAVING MAIN GMM")
    filename = 'caracterizaciones/' + region_name + '/' + region_name+"_"+ str(N)+ '/'
    os.makedirs(filename)

    np.savetxt( (filename + 'weights' + '.csv'), gmm.weights_, fmt='%.18e', delimiter=';', newline='\n', header=('GMM_converged, all '+region_name  + 'datasets  '))
    np.savetxt( (filename + 'means' + '.csv'), gmm.means_, fmt='%.18e', delimiter=';', newline='\n', header=('GMM_converged, all '+region_name  + 'datasets  '))
    np.savetxt( (filename + 'covars' + '.csv'), gmm.covars_, fmt='%.18e', delimiter=';', newline='\n', header=('GMM_converged, all '+region_name  + 'datasets  '))
    np.savetxt( (filename + 'converged' + '.csv'), converged_arr , fmt='%.18e', delimiter=';', newline='\n', header=('GMM_converged, all '+region_name  + 'datasets  '))

    return

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



    #def leerGMM( region_name, n_comp ):
    #gmmFirstLanguaje = leerGMM( firstLanguaje.name, g)
    #gmmSecondLanguaje = leerGMM( secondLanguaje.name, g)
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
regions = LC.get_MFFC_for_regions(regions)




main(0,1,30,regions)
main(0,1,30,regions)
main(0,1,30,regions)

main(1,0,30,regions)
main(1,0,30,regions)
main(1,0,30,regions)






"""
Propietary ALGORITHM
g = 1024

main(0,1,30,regions)   #Chinese0   English30
main(0,1,30,regions)   #Chinese14  English16
main(0,1,30,regions)   #Chinese0   English30
main(1,0,30,regions)   # English67  Chinese0
main(1,0,30,regions)   # English67  Chinese0
main(1,0,30,regions)   #English0  Chinese67

g = 2048

main(0,1,50,regions)   #Chinese0 English10
main(0,1,50,regions)   #Chinese0 English10
main(0,1,50,regions)   #Chinese0 English10



"""
