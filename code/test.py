from code.modules import load_regions as csv
from code.modules import learning_characterization as LC
from datetime import datetime


class testFile:
    def __init__(self, region, mfcc, lpc ):
        self.region = region 
        self.mfcc = mfcc
        self.lpc = lpc
        self.lpc_prob = []
        self.mfcc_prob = []
        self.mfcc_vector_pred = []
        self.lpc_vector_pred = []
                
start = datetime.now()                
                
regions = csv.get_regions_from_csv("test/audiosTesteo.csv")
regions= LC.generate_MFFC_LPC_region_list(regions , False)
testFiles = []
#map region ------> testFile
for reg in regions:
     tf =  testFile(reg.name,reg.mfcc_concatenation,reg.lpc_concatenation)
     testFiles.append(tf)
           
testFiles = LC.gmmCalculateProb( testFiles )

""" PRINT """
for testFile in testFiles:
          print("-----"+testFile.region + "-----" )
          
          print("MFCC")
          for prob in testFile.mfcc_vector_pred:
              print(str(prob.region)+":  votes: " + str(prob.votes) +"   prob: " +str(prob.prob) )
          print("")
          print("LPC")
          
          for prob in testFile.lpc_vector_pred:
              print(str(prob.region)+":  votes: " + str(prob.votes) +"   prob: " +str(prob.prob) )
          print("")
          print("")
          print("")
    
    
print (str( '{}'.format(datetime.now() - start ) ))             
         