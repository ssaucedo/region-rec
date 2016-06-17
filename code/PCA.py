import matplotlib.pyplot as plt
import numpy as np
from code.modules import load_regions as csv
from code.modules import learning_characterization as LC
from sklearn.decomposition import PCA
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier


""" classifiers:
 from sklearn.tree import DecisionTreeClassifier --------> check n_components
 from sklearn.naive_bayes import GaussianNB
 from sklearn.neighbors import KNeighborsClassifier
 from sklearn.svm import SVC ------> quadratic complexity, 
 from sklearn.neighbors import KNeighborsClassifier


"""


class Region:
    def __init__(self, name):
        self.name = name
        self.wav_files_id_list = []
        self.mfcc_concatenation = []
        self.lpc_concatenation = []
    def add_audio_id(self,id):
        self.wav_files_id_list.append(id)    
    def set_wav_files_id_list(self, list):
        self.wav_files_id_list = list    
    def set_mfcc_concatenation(self, mfcc_con):
        self.mfcc_concatenation = mfcc_con
    def set_lpc_concatenation(self, lpc_con):
        self.lpc_concatenation = lpc_con


carpetaMFCC = 'aprendizaje/MFCC'
carpetaGMM = 'aprendizaje/GMM'

#########################################################################################################
regions = csv.get_regions_from_csv('aprendizaje/audiosAprendizaje.csv')
for region in regions:
    print(region.name)
    region.mfcc_concatenation =  np.loadtxt( (carpetaMFCC + '/' + region.name + '/' + 'MFCC_' + "directo" + '_' + region.name + '.csv' ), delimiter=';',  skiprows=1, ndmin=0) 
    region.lpc_concatenation =  np.loadtxt( (carpetaMFCC + '/' + region.name + '/' + 'MFCC_' + "lpc" + '_' + region.name + '.csv' ), delimiter=';',  skiprows=1, ndmin=0) 

fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(12,6))

mfcc1 = regions[0].mfcc_concatenation
mfcc2 = regions[1].mfcc_concatenation
conc = np.concatenate((mfcc1, mfcc2))


y1 = np.arange(mfcc1.shape[0])
y2 = np.arange(mfcc2.shape[0])
y1.fill(1) # -------------------------> 1 = buenosaires
y2.fill(2) # -------------------------> 2 = cordoba
y1.reshape(1,-1).T
y2.reshape(1,-1).T
y = np.concatenate((y1, y2))


#classifier = DecisionTreeClassifier()

#classifier = GaussianNB()

#classifier = SVC(gamma=2, C=1)

classif = 10
classifier = KNeighborsClassifier(classif)
# DEFINE n_components
print(classif)
nco = 4
pca = PCA(n_components = nco)
print(nco)
X_r = pca.fit_transform(conc)   # No dimension reduction
classifier.fit(X_r, y)


#############################################################################

regions_Test_init = csv.get_regions_from_csv("test/audiosTesteo.csv")
region_Test_list = []




# This loop generates a list a test_files to be handled by generate_MFFC_LPC_region_list()
nro_ids = len(regions_Test_init[0].wav_files_id_list)
for x in range (0,nro_ids):
      tupleReg = []
      regBAS = Region(regions_Test_init[0].name)
      regBAS.set_wav_files_id_list([regions_Test_init[0].wav_files_id_list[x]])
      regCBA = Region(regions_Test_init[1].name)
      regCBA.set_wav_files_id_list([regions_Test_init[1].wav_files_id_list[x]])
      tupleReg.append(regBAS) 
      tupleReg.append(regCBA)
      region_Test_list.append(tupleReg)
      

for reg in region_Test_list:            
      regions_Test = LC.generate_MFFC_LPC_region_list(reg , False)
      cordoba = 0
      buenos_aires = 0
      # CALCULO COMO PREDICE MFCC DE BUENOS AIRES
      for x in regions_Test[0].mfcc_concatenation:
        pca_t= pca.transform(x)
        if(classifier.predict(pca_t) == 2 ):
            cordoba +=1
        else:
            buenos_aires +=1
      print("Audio  "+str(regions_Test[0].wav_files_id_list)+"  de "+regions_Test[0].name)          
      total = buenos_aires + cordoba
      print("% buenos_aires-------->  "+str(buenos_aires/float(total)))
      print("% .cordoba-------->  "+str(cordoba/float(total)))
    
      cordoba = 0
      buenos_aires = 0
      # CALCULO COMO PREDICE MFCC DE CORDOBA
      for x in regions_Test[1].mfcc_concatenation:
        pca_t= pca.transform(x)
        if(classifier.predict(pca_t) == 2 ):
            cordoba +=1
        else:
            buenos_aires +=1
            
      print("Audio  "+str(regions_Test[1].wav_files_id_list)+"de "+regions_Test[1].name)          
      total = buenos_aires + cordoba
      print("% buenos_aires-------->  "+str(buenos_aires/float(total)))
      print("% cordoba-------->  "+str(cordoba/float(total)))
      print("")
      print("+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")




