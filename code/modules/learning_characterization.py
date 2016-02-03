# -*- coding: utf-8 -*-
## Libreria aprendizaje.py
## Funciones para realizar las diferentes funciones que terminan en el aprendizaje (MFFC, LPC, GMM)
  
# Importo librerias del sistema
import numpy as np
import warnings
import scipy.signal as scipy
import scipy.fftpack as scyfft
  
from sklearn.externals.six.moves import xrange
from sklearn.mixture import GMM
  
from code.modules import base
from code.modules import espectro
from code.modules import cargaAudio as ca
from code.modules import filters as fl
     
# Ignora DeprecationWarning
warnings.filterwarnings("ignore", category=DeprecationWarning)
  
# Defino provincias a levantar (mismo nombre que carpetas)
nombreProvincia = ['buenosaires', 'cordoba', 'entrerios', 'mendoza', 'santiago']
  
# Le asigno una etiqueta a cada provincia para referencia
etiquetaProvincia = { 'buenosaires': 0, 'cordoba': 1, 'entrerios': 2, 'mendoza': 3, 'santiago': 4 }
  
# Carpetas de archivos MFCC y GMM
carpetaMFCC = 'aprendizaje/MFCC'
carpetaGMM = 'aprendizaje/GMM'
  
# Constantes para LPC
idx = 0
SW2d = []
T = []
p = 12 # orden del filtro lineal
  
# Constantes para MFCC
ventana = "hamming"
L = 256
M = 85
nroFiltrosMEL = 13
 
powerRelation = 0.2 #------> Relacion potencia promedio - potencia ventana. Se procesan aquellas ventanas con potencia mayor al [powerRelation * Potencia-promedio]
  
  
def save_info( data, name, id_list, mode ): 
    print("SAVING: "+name + "  MODE: "+ mode)
    np.savetxt( (carpetaMFCC + '/' + name + '/' + 'MFCC_' + mode + '_' + name + '.csv' ), data, fmt='%.18e', delimiter=';', newline='\n', header=('REGION: ' + name + ', MODE: ' + mode +', PROCESSED IDs:   '+" ".join(str(x) for x in id_list)), footer='', comments='# ')
    return
  
# Levanta MFCC de archivo CSV   
def leerMFCC( provincia, camino ):
    
    mfcc = np.loadtxt( (carpetaMFCC + '/' + provincia + '/' + 'MFCC_' + camino + '_' + provincia + '.csv' ), delimiter=';',  skiprows=1, ndmin=0) # Evito la primera row por contener el comentario
      
    return mfcc
  
# Guarda GMM en varios archivos CSV    
def saveGMM( gmm, region_name, mode ):
      
    converged_arr = np.array( [gmm.converged_] )
      
    filename = carpetaGMM + '/' + region_name + '/' + mode + '/' + 'GMM_' + region_name + '_' + mode + '_'
      
    print('Saving GMM (' + mode + ') ' + region_name + ' in ' + carpetaGMM + '/' + region_name + '/' + mode + '/' )
      
    np.savetxt( (filename + 'weights' + '.csv'), gmm.weights_, fmt='%.18e', delimiter=';', newline='\n', header=('GMM_weights from ' + region_name + ' MODE: ' + mode + '.'))
    np.savetxt( (filename + 'means' + '.csv'), gmm.means_, fmt='%.18e', delimiter=';', newline='\n', header=('GMM_means from ' + region_name + ' MODE: ' + mode + '.'))
    np.savetxt( (filename + 'covars' + '.csv'), gmm.covars_, fmt='%.18e', delimiter=';', newline='\n', header=('GMM_covars from ' + region_name + ' MODE: ' + mode + '.'))
    np.savetxt( (filename + 'converged' + '.csv'), converged_arr , fmt='%.18e', delimiter=';', newline='\n', header=('GMM_converged from ' + region_name + ' MODE: ' + mode + '.'))
  
    return
      
# Levanta los parametros de GMM de varios archivos CSV y los devuelve como una unica variable
def leerGMM( provincia, camino ):
      
    gmm = GMM(n_components= 18 , covariance_type = 'diag')
      
    filename = carpetaGMM + '/' + provincia + '/' + camino + '/' + 'GMM_' + provincia + '_' + camino + '_'
      
    gmm.weights_ = np.loadtxt( (filename + 'weights' + '.csv'), delimiter=';',  skiprows=1, ndmin=0) # Evito la primera row por contener el comentario
    gmm.means_ = np.loadtxt( (filename + 'means' + '.csv'), delimiter=';',  skiprows=1, ndmin=0)
    gmm.covars_ = np.loadtxt( (filename + 'covars' + '.csv'), delimiter=';',  skiprows=1, ndmin=0)
    gmm.converged_ = np.loadtxt( (filename + 'converged' + '.csv'), delimiter=';',  skiprows=1, ndmin=0)
      
    return gmm
  
# Compara MFCC pasado con GMM ya analizado (funcion para testeo)    
def gmmCompararAprendido( testMFCC, provinciaAprendida, camino ):
  
    gmm = leerGMM( provinciaAprendida, camino)
      
    prediccion = gmm.predict_proba(testMFCC)
    weight = gmm.weights_.reshape(1, -1).T
      
    probabilidad = np.dot(prediccion, weight)
      
    return probabilidad
      
    

class Prob:
    def __init__(self, region, vector_pred):
        self.region = region 
        self.prob = None
        self.vector_pred = vector_pred
        self.votes = 0;
              

def gmmCalculateProb(testFiles):
      region_names = [f.region for f in testFiles]
      for tf in testFiles:
           for region in region_names:
              # gererates prob_vertor for each region
              # with mfcc params
              gmm = leerGMM( region, 'directo')
              prediccion = gmm.predict_proba(tf.mfcc)
              weight = gmm.weights_.reshape(1, -1).T            
              tf.mfcc_vector_pred.append(Prob(region,np.dot(prediccion, weight)))
              
              # with mfcc params
              gmm = leerGMM( region, 'lpc')
              prediccion = gmm.predict_proba(tf.lpc)
              weight = gmm.weights_.reshape(1, -1).T            
              tf.lpc_vector_pred.append(Prob(region,np.dot(prediccion, weight)))
      testFiles = calculate_votes(testFiles)
      testFiles = calculate_prob(testFiles)
      
      return testFiles

def calculate_votes(testFiles):
           for testFile in testFiles:
               mfcc_vector_list = testFile.mfcc_vector_pred
               nro_in_vector_pred = len(mfcc_vector_list[0].vector_pred)-1
               for i in xrange(0,nro_in_vector_pred):
                    v_pred = [v.vector_pred[i] for v in mfcc_vector_list] #tomo cada valor de la lista de vector_prob 
                    vote_index = v_pred.index(max(v_pred))
                    testFile.mfcc_vector_pred[vote_index].votes = testFile.mfcc_vector_pred[vote_index].votes+1
               lpc_vector_list = testFile.lpc_vector_pred
               nro_in_vector_pred = len(lpc_vector_list[0].vector_pred)-1
               for i in xrange(0,nro_in_vector_pred):
                    v_pred = [v.vector_pred[i] for v in lpc_vector_list] #tomo cada valor de la lista de vector_prob 
                    vote_index = v_pred.index(max(v_pred))
                    testFile.lpc_vector_pred[vote_index].votes = testFile.lpc_vector_pred[vote_index].votes+1
           return testFiles

def calculate_prob(testFiles):
           for testFile in testFiles:
               total_mfcc_votes = 0
               total_lpc_votes = 0
               for i in xrange(0,len(testFile.mfcc_vector_pred)):
                    total_mfcc_votes = total_mfcc_votes + testFile.mfcc_vector_pred[i].votes
               for i in xrange(0,len(testFile.lpc_vector_pred)):
                    total_lpc_votes = total_lpc_votes + testFile.lpc_vector_pred[i].votes
               for i in xrange(0,len(testFile.mfcc_vector_pred)):
                    testFile.mfcc_vector_pred[i].prob = float(testFile.mfcc_vector_pred[i].votes) / total_mfcc_votes
               for i in xrange(0,len(testFile.lpc_vector_pred)):
                    testFile.lpc_vector_pred[i].prob = float(testFile.lpc_vector_pred[i].votes) / total_lpc_votes
           return testFiles

                      
  
def generate_GMM_region_list( regions, mode ):
    for region in regions:
        if(mode == "directo"):
           calculoGMM( region.mfcc_concatenation, region.name, mode)      
        elif(mode == "lpc"):
           calculoGMM( region.lpc_concatenation, region.name, mode)                 
    return
      
# Funcion para calcular GMM en base a un MFCC y lo guarda en archivo CSV
def calculoGMM( mfcc_concatenation, region_name, mode ):
      
    gmm = GMM(n_components= 18 , covariance_type = 'diag')
    gmm.fit(mfcc_concatenation)
    saveGMM( gmm, region_name, mode )
    
    return
          
          
# """ FUNCIONES A LLAMAR DESDE EL MAIN """
  
def generate_MFFC_LPC_region_list( regions , sav_data ):
    for region in  regions :
        mfcc = []
        lpc = []
        for file_id in region.wav_files_id_list:
           if (len(mfcc) == 0):
                mfcc , lpc = generate_MFCC_LPC(region.name , file_id)
           else:               
                mfcc_c , lpc_c = generate_MFCC_LPC(region.name , file_id)
                mfcc = np.concatenate((mfcc, mfcc_c))
                lpc = np.concatenate((lpc, lpc_c))
        if(sav_data):
           save_info( mfcc, region.name, region.wav_files_id_list, 'directo' )
           save_info( lpc, region.name, region.wav_files_id_list, 'lpc' )
        region.set_lpc_concatenation(lpc)
        region.set_mfcc_concatenation(mfcc)
    return  regions
    
def generate_MFCC_LPC(name , file_id):
     print(name+"    "+ file_id)
     Fs , dat = ca.load_wav_file(name , file_id)
     dat = fl.filtroPreEnfasis(dat)
     dat = fl.filtroFIR(dat, Fs)
     
     # CalculateMFCC
     WDFFT, Fs = getFilteredWindowedDFFT(dat, Fs)
     MF = getMF(WDFFT , nroFiltrosMEL, Fs)
     MFCC = getMFCC(MF)
     
     #CalculateLPC
     LPC = getFilteredLPC(dat , Fs)
     WDFFT_LPC, Fs = getWindowedDFFT(LPC , Fs)
     MF_LPC = getMF(WDFFT_LPC , nroFiltrosMEL ,Fs)
     MFCC_LPC = getMFCC(MF_LPC)
     return MFCC , MFCC_LPC
  

    
"""
FUNCIONES  MFFC 
"""
# FUNCIONES PARA GENERAR MFCC A PARTIR DE UN AUDIO---->  getWindowedDFFT , getMF , getMFCC.
    
#Recibe el nombre del archivo a leer ,devuelve 2D np.array donde cada linea corresponde 
# a la DFFT de una ventana.   WDFFT.shape(n , 513)  ----> n depende de la cantidad de ventanas.
# Filtered----> Implica que la data es filtrada en funcion de su relacion con la potencia promedio , 
# ****corresponde al analisis de MFFC directo , no tiene sentido filtrar cuando la data corresponde a la formante****
def getFilteredWindowedDFFT(data , Fs):
     N = len(data);
     WDFFT = [] 
     i = 0
     idx = 0
     P = 0
     wn = scipy.get_window(ventana,L)
     while idx < N-L-1:
          y = data[idx:idx+L] * wn
          P = P + espectro.powerOf(y,L)
          idx = idx + M
          i = i +1
     P = P/i  #-------->    Potencia por ventana *promedio    
     idx = 0
     while idx < N-L-1:
          y = data[idx:idx+L] * wn
          Pv =espectro.powerOf(y,L)
          if (Pv > P* powerRelation):
              f, Y = espectro.spectrum(y,Fs)
              WDFFT.append(Y)       
          idx = idx + M
     return np.asarray(WDFFT),Fs
     
"""
FUNCIONES  LPC 
"""

def getWindowedDFFT(data , Fs):
     N = len(data);
     WDFFT = [] 
     idx = 0
     wn = scipy.get_window(ventana,L)
     idx = 0
     while idx < N-L-1:
          y = data[idx:idx+L] * wn
          f , Y = espectro.spectrum(y,Fs)
          WDFFT.append(Y)       
          idx = idx + M
     return np.asarray(WDFFT),Fs  
    
        
def getFilteredLPC(data , Fs):
     N = len(data);
     WLPC = [] 
     idx = 0
     wn = scipy.get_window(ventana,L)
     while idx < N-L-1:
          b = np.ndarray(1)
          b[0] = 1
          y = data[idx:idx+L] * wn
          a,Rmat,r = computeLPC(y)
          b = np.concatenate((b,-a))
          r = np.roots(b)
          r =r[np.absolute(r)> 0.8]
          A = np.angle(r)
          A = A[ A > 0]
          A = A[A < np.pi]
          AHz = np.int32(A * np.float32(Fs) / (2*np.pi))
          idx = idx + M
          sortedFreqs = np.sort(AHz)
          if (sortedFreqs.shape != ()): WLPC.append(sortedFreqs[0])
     return WLPC    
        
  
def computeLPC(s):
    L = len(s)
    Rmat = np.zeros((p,p))
    r = np.zeros(p)
    R = np.zeros(p+1) 
    for n in range(0,p+1):
        R[n] = 0.
        for k in range(0,L-n-1):
            R[n] = R[n] + s[k]*s[k+n]
    r = R[-p:] 
    for i in range(0,p):
        for j in range(0,p):
            Rmat[i,j] = R[abs(i-j)]
    # ahora calculo los coeficientes
    a = np.linalg.lstsq(Rmat,r)
    return a[0],Rmat,r      
    
    
""" MFFC-LPC """

#Recibe el arreglo WDFFT , es decir la Matriz que contiene las DFFT de cada ventana de el audio analizado 
# devuelve la Matriz MF ( ver documentacion). A cada linea WDFFT , se le aplicara el conjunto de filtros 
# a cada combinacion se le calculara la potencia y se guardara en la matriz MF.
# Asi el numero de lineas sera igual al de WDFFT , y cada columna correspondera a la potencia para la señal por 
# el filtro correspondiente.  ---> MF.shape = (WDFFT.shape[0],nroFiltrosMEL)    
def getMF(WDFFT , nroFiltrosMEL ,Fs):    
     filterParam = (WDFFT[0].shape[0]-1)*2 
     fb = base.get_filterbanks(nroFiltrosMEL,filterParam,Fs)
     MFArray = np.zeros((WDFFT.shape[0], nroFiltrosMEL))
     fl = np.asarray(list(range(0, WDFFT[0].shape[0])))
     #for window in WDFFT:
     for i in xrange(0,WDFFT.shape[0]):
         for j in xrange(0,nroFiltrosMEL):
              fl1= fb[j,fl]
              MF = fl1 * WDFFT[i]   #---------> MF = Ventana * Filtro de Mel
              MFp =  espectro.powerOf(MF,filterParam/2) #---------> MFp = energia de MF 
              np.put(MFArray[i],[j],MFp) #---------> Agrego la potencia MFp en la posicion que corresponde.
     return MFArray
     
# Recibe la Matriz MF y realiza la logica para obtener los MFCC.
# Primero calcula el logaritmo (Cepstrum) y luego el coseno inverso.
# Realiza esta operacion para cada ventana ( o linea)      
def getMFCC(MF):
     cm = np.log10(MF)  #Cepstrum
     MFCC = []
     for c in cm:
         mffcCoef  = scyfft.idct(c)   #Coseno inverso
         MFCC.append(mffcCoef) 
     MFCC = np.array(MFCC)
     return MFCC
  
  
   

        