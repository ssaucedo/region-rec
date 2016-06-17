# -*- coding: utf-8 -*-
import numpy as np
from sklearn.mixture import GMM
from code.modules import load_regions as csv
from code.modules import learning_characterization as LC

carpetaGMM = 'aprendizaje/GMM'


class Prob:
    def __init__(self, region, vector_pred):
        self.region = region 
        self.prob = None
        self.vector_pred = vector_pred
        self.votes = 0;

def leerGMM( provincia, camino ):
      
    gmm = GMM(n_components= 18 , covariance_type = 'diag')
      
    filename = carpetaGMM + '/' + provincia + '/' + camino + '/' + 'GMM_' + provincia + '_' + camino + '_'
      
    gmm.weights_ = np.loadtxt( (filename + 'weights' + '.csv'), delimiter=';',  skiprows=1, ndmin=0) # Evito la primera row por contener el comentario
    gmm.means_ = np.loadtxt( (filename + 'means' + '.csv'), delimiter=';',  skiprows=1, ndmin=0)
    gmm.covars_ = np.loadtxt( (filename + 'covars' + '.csv'), delimiter=';',  skiprows=1, ndmin=0)
    gmm.converged_ = np.loadtxt( (filename + 'converged' + '.csv'), delimiter=';',  skiprows=1, ndmin=0)
      
    return gmm
  


def gmmCalculateProb(testFiles):
              gmm1 = leerGMM( testFiles[0].region, 'directo')
              gmm2 = leerGMM( testFiles[1].region, 'directo')
              
              prediccion1 = gmm1.predict_proba(testFiles[0].mfcc)
              prediccion2 = gmm2.predict_proba(testFiles[0].mfcc)
              
              weight1 = gmm1.weights_.reshape(1, -1).T            
              weight2 = gmm2.weights_.reshape(1, -1).T            
              
              clusters_count1 = np.zeros(18) 
              clusters_count2 = np.zeros(18)
              
              testFiles[0].mfcc_vector_pred.append(Prob(testFiles[0].region,np.dot(prediccion1, weight1)))
              testFiles[0].mfcc_vector_pred.append(Prob(testFiles[1].region,np.dot(prediccion2, weight2)))
              testFile = testFiles[0]
              print("TestData Lenght "+ str(len(prediccion1)))
              mfcc_vector_list = testFile.mfcc_vector_pred
              nro_in_vector_pred = len(mfcc_vector_list[0].vector_pred)-1
              for i in xrange(0,nro_in_vector_pred):
                       v_pred = [v.vector_pred[i] for v in mfcc_vector_list] #tomo cada valor de la lista de vector_prob 
                       vote_index = v_pred.index(max(v_pred))
                       testFile.mfcc_vector_pred[vote_index].votes = testFile.mfcc_vector_pred[vote_index].votes+1
              
              print("votos cordoba--------------->"+str(testFile.mfcc_vector_pred[0].votes))
              print("votos buenos aires------------->"+str(testFile.mfcc_vector_pred[1].votes))
              total = testFile.mfcc_vector_pred[0].votes + testFile.mfcc_vector_pred[1].votes
              print("total: "+ str(total))
              print("% cordoba--------------->"+str(testFile.mfcc_vector_pred[0].votes / float(total)))
              print("% buenos aires------------->"+str(testFile.mfcc_vector_pred[1].votes / float(total)))
                 
              testFile.mfcc_vector_pred[0].votes = 0
              testFile.mfcc_vector_pred[1].votes = 0
              testFiles[0].mfcc_vector_pred = []
                 
                
##############################################################################################
              for p in prediccion1:
                      clusters_count1[np.argmax(p)] = clusters_count1[np.argmax(p)] + 1
              for p in prediccion2:
                      clusters_count2[np.argmax(p)] = clusters_count2[np.argmax(p)] + 1
               
              
              for L in range (1,9):
                  
                 r = []
                 max_list_clus1 = clusters_count1.argsort()[-L:][::-1]
                 max_list_clus2 = clusters_count2.argsort()[-L:][::-1]
              
                 for x in range (0,len(prediccion1)):
                     if ( np.in1d((np.argmax(prediccion1[x])),max_list_clus1) and
                       (np.in1d((np.argmax(prediccion2[x])),max_list_clus2))):
                           r.append(x)
                 del x
                 data_indexes = set(r)
                 print("Filtering deleting "+str(L))
                 prediccion1_del = [v for i, v in enumerate(prediccion1) if i not in data_indexes]
                 prediccion2_del = [v for i, v in enumerate(prediccion2) if i not in data_indexes]
                 testFiles[0].mfcc_vector_pred.append(Prob(testFiles[0].region,np.dot(prediccion1_del, weight1)))
                 testFiles[0].mfcc_vector_pred.append(Prob(testFiles[1].region,np.dot(prediccion2_del, weight2)))
                 testFile = testFiles[0]
                 print("Initial lenght "+ str(len(prediccion1)))
                 print("Final lenght "+ str(len(prediccion1_del)))
                 mfcc_vector_list = testFile.mfcc_vector_pred
                 
                 nro_in_vector_pred = len(mfcc_vector_list[0].vector_pred)-1
                 for i in xrange(0,nro_in_vector_pred):
                       v_pred = [v.vector_pred[i] for v in mfcc_vector_list] #tomo cada valor de la lista de vector_prob 
                       vote_index = v_pred.index(max(v_pred))
                       testFile.mfcc_vector_pred[vote_index].votes = testFile.mfcc_vector_pred[vote_index].votes+1
                 i = 0
                 print("votos cordoba--------------->"+str(testFile.mfcc_vector_pred[0].votes))
                 print("votos buenos aires------------->"+str(testFile.mfcc_vector_pred[1].votes))
                 total = testFile.mfcc_vector_pred[0].votes + testFile.mfcc_vector_pred[1].votes
                 print("total: "+ str(total))
                 print("% cordoba--------------->"+str(testFile.mfcc_vector_pred[0].votes / float(total)))
                 print("% buenos aires------------->"+str(testFile.mfcc_vector_pred[1].votes / float(total)))
                 print("")
                 print("")
                 print("#####################################################")
                                  
                 testFile.mfcc_vector_pred[0].votes = 0
                 testFile.mfcc_vector_pred[1].votes = 0
                 testFiles[0].mfcc_vector_pred = []

              return testFile



             


class testFile:
    def __init__(self, region, mfcc, lpc ):
        self.region = region 
        self.mfcc = mfcc
        self.lpc = lpc
        self.lpc_prob = []
        self.mfcc_prob = []
        self.mfcc_vector_pred = []
        self.lpc_vector_pred = []
                
regions = csv.get_regions_from_csv("test/audiosTesteo.csv")
regions= LC.generate_MFFC_LPC_region_list(regions , False)
testFiles = []
for reg in regions:
     tf =  testFile(reg.name,reg.mfcc_concatenation,reg.lpc_concatenation)
     testFiles.append(tf)
           
t = gmmCalculateProb( testFiles )
