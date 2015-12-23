# -*- coding: utf-8 -*-
## Libreria aprendizaje.py
## Funciones para realizar las diferentes funciones que terminan en el aprendizaje (MFFC, LPC, GMM)
  
# Importo librerias del sistema
import numpy as np
import warnings
import scipy.signal as scipy
import scipy.fftpack as scyfft
  
# Importo librerias de aprendizaje (sklearn)
from sklearn.externals.six.moves import xrange
from sklearn.mixture import GMM
  
# Importo librerias ajenas
# Bajada de https://github.com/jameslyons/python_speech_features
#from codigo.features import fbank
from codigo.features import base
  
# Importo librerias propias
from codigo import parametrosCSV as csv
from codigo import espectro
  
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
  
# Funcion para seleccionar los audios para aprendizaje. Toma la variable que contiene todos los audios y solo deja los deseados (definidos en CSV)
def audiosAprendizaje( datosProvincia ):
      
    numAudiosAprendizaje = csv.numAudiosParaAprendizaje()
      
    datosProvinciaAprendizaje = [None] * len(numAudiosAprendizaje) # Inicializo en None, necesito para despues referirme al numero de index
      
    for provincia in range( len(numAudiosAprendizaje) ):
        datTemporal = []
        for audio in range( len(numAudiosAprendizaje[provincia]) ):
            datTemporal.append( datosProvincia[provincia][ numAudiosAprendizaje[provincia][audio] - 1 ] ) # Cargo en el vector el audio especificado en el csv
                                                                                            # Resto 1 porque los registros en la pc van de 1-20 mientras en los vectores van de 0-19
        # Cierro aqui "for audio"
        datosProvinciaAprendizaje[provincia] = datTemporal
        del datTemporal
      
    # Cierro aqui "for provincia"
    del audio
    del provincia
                  
    return datosProvinciaAprendizaje
  
# Guarda los MFCC en un archivo CSV    
def guardarMFCC( mfcc, provincia, camino ): # Le paso el string de la provincia y el string del camino (MFCC directo o pasado antes por LPC)
      
    numAudiosAprendizaje = csv.numAudiosParaAprendizaje()
      
    print('Guardado MFCC (' + camino + ') de ' + provincia + ' en ' + carpetaMFCC + '/' + provincia +'/')
    np.savetxt( (carpetaMFCC + '/' + provincia + '/' + 'MFCC_' + camino + '_' + provincia + '.csv' ), mfcc, fmt='%.18e', delimiter=';', newline='\n', header=('MFCC de la provincia de ' + provincia + ', mediante el camino ' + camino + '. Audios analizados: ' + str( numAudiosAprendizaje[ etiquetaProvincia[provincia]] ) + '. En total: ' + str(len( numAudiosAprendizaje[ etiquetaProvincia[provincia]])) + ' audios.' ), footer='', comments='# ')
  
    return
  
# Levanta MFCC de archivo CSV   
def leerMFCC( provincia, camino ):
      
    mfcc = np.loadtxt( (carpetaMFCC + '/' + provincia + '/' + 'MFCC_' + camino + '_' + provincia + '.csv' ), delimiter=';',  skiprows=1, ndmin=0) # Evito la primera row por contener el comentario
      
    return mfcc
  
# Guarda GMM en varios archivos CSV    
def guardarGMM( gmm, provincia, camino ):
      
    converged_arr = np.array( [gmm.converged_] )
      
    filename = carpetaGMM + '/' + provincia + '/' + camino + '/' + 'GMM_' + provincia + '_' + camino + '_'
      
    print('Guardado GMM (' + camino + ') de ' + provincia + ' en ' + carpetaGMM + '/' + provincia + '/' + camino + '/' )
      
    np.savetxt( (filename + 'weights' + '.csv'), gmm.weights_, fmt='%.18e', delimiter=';', newline='\n', header=('GMM_weights de la provincia ' + provincia + ' por el camino ' + camino + '.'))
    np.savetxt( (filename + 'means' + '.csv'), gmm.means_, fmt='%.18e', delimiter=';', newline='\n', header=('GMM_means de la provincia ' + provincia + ' por el camino ' + camino + '.'))
    np.savetxt( (filename + 'covars' + '.csv'), gmm.covars_, fmt='%.18e', delimiter=';', newline='\n', header=('GMM_covars de la provincia ' + provincia + ' por el camino ' + camino + '.'))
    np.savetxt( (filename + 'converged' + '.csv'), converged_arr , fmt='%.18e', delimiter=';', newline='\n', header=('GMM_converged de la provincia ' + provincia + ' por el camino ' + camino + '.'))
  
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
      
# Calcular la probabilidad de un MFCC de pertenecer a una determinada provincia (funcion para testeo)
def gmmCalcularProbabilidad( testMFCC, camino ):    
  
    numAudiosAprendizaje = csv.numAudiosParaAprendizaje()
      
    probabilidadesTest = [None] * len(numAudiosAprendizaje) # Inicializo en None, necesito para despues referirme al numero de index
     
    probabilidadesTest[ etiquetaProvincia['buenosaires'] ] = gmmCompararAprendido( testMFCC, 'buenosaires', camino )
    probabilidadesTest[ etiquetaProvincia['cordoba'] ] = gmmCompararAprendido( testMFCC, 'cordoba', camino )
    probabilidadesTest[ etiquetaProvincia['entrerios'] ] = gmmCompararAprendido( testMFCC, 'entrerios', camino )
    probabilidadesTest[ etiquetaProvincia['mendoza'] ] = gmmCompararAprendido( testMFCC, 'mendoza', camino )
    probabilidadesTest[ etiquetaProvincia['santiago'] ] = gmmCompararAprendido( testMFCC, 'santiago', camino )
      
    buenosairesVotos = 0
    cordobaVotos = 0
    entreriosVotos = 0
    mendozaVotos = 0
    santiagoVotos = 0
      
    for cont in xrange( 0, probabilidadesTest[0].shape[0] ):  # Todas las posiciones deberian tener el mismo largo
          
        if ( (probabilidadesTest[etiquetaProvincia['buenosaires']][cont] > probabilidadesTest[etiquetaProvincia['cordoba']][cont]) and (probabilidadesTest[etiquetaProvincia['buenosaires']][cont] > probabilidadesTest[etiquetaProvincia['entrerios']][cont]) and (probabilidadesTest[etiquetaProvincia['buenosaires']][cont] > probabilidadesTest[etiquetaProvincia['mendoza']][cont]) and (probabilidadesTest[etiquetaProvincia['buenosaires']][cont] > probabilidadesTest[etiquetaProvincia['santiago']][cont]) ):
            buenosairesVotos += 1
        elif ( (probabilidadesTest[etiquetaProvincia['cordoba']][cont] > probabilidadesTest[etiquetaProvincia['buenosaires']][cont]) and (probabilidadesTest[etiquetaProvincia['cordoba']][cont] > probabilidadesTest[etiquetaProvincia['entrerios']][cont]) and (probabilidadesTest[etiquetaProvincia['cordoba']][cont] > probabilidadesTest[etiquetaProvincia['mendoza']][cont]) and (probabilidadesTest[etiquetaProvincia['cordoba']][cont] > probabilidadesTest[etiquetaProvincia['santiago']][cont]) ):
            cordobaVotos += 1
        elif ( (probabilidadesTest[etiquetaProvincia['entrerios']][cont] > probabilidadesTest[etiquetaProvincia['buenosaires']][cont]) and (probabilidadesTest[etiquetaProvincia['entrerios']][cont] > probabilidadesTest[etiquetaProvincia['cordoba']][cont]) and (probabilidadesTest[etiquetaProvincia['entrerios']][cont] > probabilidadesTest[etiquetaProvincia['mendoza']][cont]) and (probabilidadesTest[etiquetaProvincia['entrerios']][cont] > probabilidadesTest[etiquetaProvincia['santiago']][cont]) ):
            entreriosVotos += 1
        elif ( (probabilidadesTest[etiquetaProvincia['mendoza']][cont] > probabilidadesTest[etiquetaProvincia['buenosaires']][cont]) and (probabilidadesTest[etiquetaProvincia['mendoza']][cont] > probabilidadesTest[etiquetaProvincia['cordoba']][cont]) and (probabilidadesTest[etiquetaProvincia['mendoza']][cont] > probabilidadesTest[etiquetaProvincia['entrerios']][cont]) and (probabilidadesTest[etiquetaProvincia['mendoza']][cont] > probabilidadesTest[etiquetaProvincia['santiago']][cont]) ):
            mendozaVotos += 1
        elif ( (probabilidadesTest[etiquetaProvincia['santiago']][cont] > probabilidadesTest[etiquetaProvincia['buenosaires']][cont]) and (probabilidadesTest[etiquetaProvincia['santiago']][cont] > probabilidadesTest[etiquetaProvincia['cordoba']][cont]) and (probabilidadesTest[etiquetaProvincia['santiago']][cont] > probabilidadesTest[etiquetaProvincia['entrerios']][cont]) and (probabilidadesTest[etiquetaProvincia['santiago']][cont] > probabilidadesTest[etiquetaProvincia['mendoza']][cont]) ):
            santiagoVotos += 1
        else:
            print('Voto no asignado')
      
      
    votosTotal = buenosairesVotos + cordobaVotos + entreriosVotos + mendozaVotos + santiagoVotos
      
    # LLevo a % los votos
    buenosairesVotos = float(buenosairesVotos) / float(votosTotal)
    cordobaVotos = float(cordobaVotos) / float(votosTotal)
    entreriosVotos = float(entreriosVotos) / float(votosTotal)
    mendozaVotos = float(mendozaVotos) / float(votosTotal)
    santiagoVotos = float(santiagoVotos) / float(votosTotal)
      
    vectorProb = [None] * len(numAudiosAprendizaje)
      
    vectorProb[ etiquetaProvincia['buenosaires'] ] = buenosairesVotos
    vectorProb[ etiquetaProvincia['cordoba'] ] = cordobaVotos
    vectorProb[ etiquetaProvincia['entrerios'] ] = entreriosVotos
    vectorProb[ etiquetaProvincia['mendoza'] ] = mendozaVotos
    vectorProb[ etiquetaProvincia['santiago'] ] = santiagoVotos
      
    return vectorProb
  
# Se le pasa una MFCC y un LPC (MFCC con LPC) y analiza probabilidades para ambos metodos
def gmmCalcularProbabilidadAmbosCaminos( testMFCC, testLPC ):
      
    vectorProb_directo = gmmCalcularProbabilidad( testMFCC, 'directo')
    vectorProb_lpc = gmmCalcularProbabilidad( testLPC, 'lpc')
      
    return vectorProb_directo, vectorProb_lpc
  
# Funcion que recibe varios MFCC de varias provicinas y los pasa de manera individual a calculoGMM    
def calculoGMMcomposer( mfccProvincia, camino ):
      
    for provincia in range( len(mfccProvincia) ):
        calculoGMM( mfccProvincia[provincia], nombreProvincia[provincia], camino)
          
    # Cierro aqui "for provincia"
    del provincia
      
    return
      
# Funcion para calcular GMM en base a un MFCC y lo guarda en archivo CSV
def calculoGMM( mfcc, provincia, camino ):
      
    gmm = GMM(n_components= 18 , covariance_type = 'diag')
    gmm.fit(mfcc)
      
    guardarGMM( gmm, provincia, camino )
      
    return
          
          
# """ FUNCIONES A LLAMAR DESDE EL MAIN """
  
# Funcion que recibe multiples
def generateLPCcomposer( datosProvincia, Fs ):
      
    lpcProvincia = [None] * len(datosProvincia) # Inicializo en None, necesito para despues referirme al numero de index
      
    for provincia in range( len(datosProvincia) ):
        datTemporal = []
        for audio in range( len(datosProvincia[provincia]) ):
            if (audio == 0):        # En la primera iteracion no uso concatenate porque falla con array nulos (none)
                datTemporal = generateLPC(datosProvincia[provincia][audio] , Fs )
            else:
                datTemporal = np.concatenate( (datTemporal, generateLPC(datosProvincia[provincia][audio] , Fs )) )
          
        # Cierro aqui "for audio"
        lpcProvincia[provincia] = datTemporal
        guardarMFCC( datTemporal, nombreProvincia[provincia], 'lpc' )
        del datTemporal
      
    # Cierro aqui "for provincia"
    del audio
    del provincia
                  
    return lpcProvincia
  
def generateLPC(datos , Fs ):
    LPC = getFilteredLPC(datos , Fs)
    LpcWDFFT, Fs = getWindowedDFFT(LPC , Fs)
    MF = getMF(LpcWDFFT , nroFiltrosMEL ,Fs)
      
    return getMFCC(MF)
      
def generateMFCCcomposer( datosProvincia, Fs ):
      
    mfccProvincia = [None] * len(datosProvincia) # Inicializo en None, necesito para despues referirme al numero de index
      
    for provincia in range( len(datosProvincia) ):
        datTemporal = []
        for audio in range( len(datosProvincia[provincia]) ):
            if (audio == 0):
                datTemporal = generateMFCC(datosProvincia[provincia][audio] , Fs )
            else:               
                datTemporal = np.concatenate( (datTemporal, generateMFCC(datosProvincia[provincia][audio] , Fs )) )
          
        # Cierro aqui "for audio"
        mfccProvincia[provincia] = datTemporal
        guardarMFCC( datTemporal, nombreProvincia[provincia], 'directo' )
        del datTemporal
      
    # Cierro aqui "for provincia"
    del audio
    del provincia
                  
    return   mfccProvincia  
      
def generateMFCC(audio , Fs):
     WDFFT, Fs = getFilteredWindowedDFFT(audio , Fs)
     MF = getMF(WDFFT , nroFiltrosMEL ,Fs)
       
     return getMFCC(MF)
     

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
          if (sortedFreqs.shape != ()): WLPC.append(sortedFreqs[1])
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