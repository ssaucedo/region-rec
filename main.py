#!/usr/bin/python3
# -*- coding: utf-8 -*-

import numpy as np
from features import mfcc
from features import logfbank
from sklearn import preprocessing

from modules import learning_characterization as LC
from sklearn.mixture import GMM
import modules.cargaAudio as ca

import os




def save_info( data, name, wav_id):

    print("SAVING: "+name)
    os.chdir("D://UADE//PFI_workspace//TestMultilingual")
    filename = 'caracterizaciones/' + name + '/MFCC_' + wav_id + '/'
    os.makedirs(filename)

    np.savetxt( (filename + 'MFCC_' + wav_id + '.csv' ), data, fmt='%.18e', delimiter=';', newline='\n', header=('REGION: ' + name + 'PROCESSED IDs:'   +wav_id), footer='', comments='# ')
    return


class Region:
    def __init__(self, name):
        self.name = name
        self.wav_files_id_list = []
        self.audio_char_list = []
    def add_audio_char(self,audio_c):
        self.audio_char_list.append(audio_c)
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

def retrieveCorporas(corpora):
    regions = []
    original_dir = os.getcwd()
    corpora_dir = os.getcwd()+"/corpora/"
    os.chdir(corpora_dir)
    original_dir = os.getcwd()
    for i in os.listdir(os.getcwd()):
          os.chdir(corpora_dir+"/"+i)
          reg = Region(i)
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

def generate_MFFC_region_list( regions):
    mapList = []
    for region in  regions :
        mapMFCC = {}
        mfcc = []
        for file_id in region.wav_files_id_list:
                mfcc  = generate_MFCC(region.name , file_id)
                save_info( mfcc, region.name, file_id)
                mapMFCC[file_id] = mfcc
        mapList.append(mapMFCC)
    return  mapList


def generate_MFCC(name , file_id):
     Fs , dat = ca.load_wav_file(name , file_id)
     MFCC = mfcc(dat,Fs)
     return MFCC


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



def main(l1,l2,nroTest,regions,gmmRecord):
    g = 1024
    mapListLoaded = LC.get_MFFC_region_map_list(regions)
    firstLanguaje = regions[l1]
    secondLanguaje = regions[l2]
    map_firstLanguaje = mapListLoaded[l1]
    map_secondLanguaje = mapListLoaded[l2]
    firstLanguajeConcatenation = []

    for k in firstLanguaje.wav_files_id_list[:nroTest]:
            if (len(firstLanguajeConcatenation) == 0):
                    firstLanguajeConcatenation = map_firstLanguaje[k]
            else:
                firstLanguajeConcatenation = np.concatenate((firstLanguajeConcatenation, map_firstLanguaje[k]))

    secondLanguajeConcatenation = []
    for k in secondLanguaje.wav_files_id_list[:nroTest]:
            if (len(secondLanguajeConcatenation) == 0):
                    secondLanguajeConcatenation = map_secondLanguaje[k]
            else:
                    secondLanguajeConcatenation = np.concatenate((secondLanguajeConcatenation, map_secondLanguaje[k]))
    #firstLanguajeConcatenation = preprocessing.scale(firstLanguajeConcatenation)
    #secondLanguajeConcatenation = preprocessing.scale(secondLanguajeConcatenation)


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


regions = retrieveCorporas('/corpora/')
mapList = generate_MFFC_region_list(regions)
gmmRecord = []

main(0,1,30,regions,gmmRecord)
main(0,1,30,regions,gmmRecord)
main(0,1,30,regions,gmmRecord)
main(0,1,30,regions,gmmRecord)
main(0,1,30,regions,gmmRecord)
main(0,1,30,regions,gmmRecord)

gmmRecord
 = []

main(1,0,30,regions,gmmRecord)
main(1,0,30,regions,gmmRecord)
main(1,0,30,regions,gmmRecord)
main(1,0,30,regions,gmmRecord)
main(1,0,30,regions,gmmRecord)
main(1,0,30,regions,gmmRecord)
gmmRecord = []


main(0,1,40,regions,gmmRecord)
main(0,1,40,regions,gmmRecord)
main(0,1,40,regions,gmmRecord)
main(0,1,40,regions,gmmRecord)
main(0,1,40,regions,gmmRecord)
main(0,1,40,regions,gmmRecord)

gmmRecord = []

main(1,0,40,regions,gmmRecord)
main(1,0,40,regions,gmmRecord)
main(1,0,40,regions,gmmRecord)
main(1,0,40,regions,gmmRecord)
main(1,0,40,regions,gmmRecord)
main(1,0,40,regions,gmmRecord)
