import matplotlib.pyplot as plt
import numpy as np
from code.modules import parametrosCSV as csv
from code.modules import learning_characterization as LC


carpetaMFCC = 'aprendizaje/MFCC'
carpetaGMM = 'aprendizaje/GMM'

regions = csv.get_regions_from_csv('aprendizaje/audiosAprendizaje.csv')

for region in regions:
    print(region.name)
    region.mfcc_concatenation =  np.loadtxt( (carpetaMFCC + '/' + region.name + '/' + 'MFCC_' + "directo" + '_' + region.name + '.csv' ), delimiter=';',  skiprows=1, ndmin=0) 
    region.lpc_concatenation =  np.loadtxt( (carpetaMFCC + '/' + region.name + '/' + 'MFCC_' + "lpc" + '_' + region.name + '.csv' ), delimiter=';',  skiprows=1, ndmin=0) 
len_mfcc = min([len(r.mfcc_concatenation) for r in regions])
len_lpc = min([len(r.lpc_concatenation) for r in regions])
for region in regions:
    region.mfcc_concatenation = region.mfcc_concatenation[:len_mfcc]
    region.lpc_concatenation = region.lpc_concatenation[:len_lpc]
                                          
LC.generate_GMM_region_list( regions , 'directo' )
LC.generate_GMM_region_list( regions , 'lpc' )


