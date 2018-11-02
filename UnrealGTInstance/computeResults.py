#!/usr/bin/env python
# -*- coding: utf-8 -*-

import pickle
import os
import shutil
import numpy as np
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
from scikitplot.metrics import plot_confusion_matrix


####################################################
##  Creates a latex table and rounds the metrics to a given number
####################################################
def toLatexTab(objList, accList, precList, recList, f1List, r):
    resultString = ''
    for i in range(0, len(objList)):
        resultString += objList[i] + ' & ' 
        resultString += str(round(accList[i], r)) + ' & ' 
        resultString += str(round(precList[i], r)) + ' & '   
        resultString += str(round(recList[i], r)) + ' & ' 
        resultString += str(round(f1List[i], r)) 
        resultString += ' \\\  \n'
    return resultString


objectList = ['AlbiHimbeerJuice','BlueCeramicIkeaMug','BlueMetalPlateWhiteSpeckles','BluePlasticBowl','BluePlasticFork','BluePlasticKnife','BluePlasticSpoon','CupEcoOrange','EdekaRedBowl','ElBrygCoffee','JaMilch','JodSalz','KelloggsCornFlakes',
    'KelloggsToppasMini','KnusperSchokoKeks','KoellnMuesliKnusperHonigNuss','LargeGreySpoon','LinuxCup','LionCerealBox','MarkenSalz','MeerSalz','MondaminPancakeMix','NesquikCereal','PfannerGruneIcetea','PfannerPfirsichIcetea','RedMetalBowlWhiteSpeckles',
    'RedMetalCupWhiteSpeckles','RedMetalPlateWhiteSpeckles','RedPlasticFork','RedPlasticKnife','RedPlasticSpoon','ReineButterMilch','SeverinPancakeMaker','SiggBottle','SlottedSpatula','SojaMilch','SpitzenReis','TomatoAlGustoBasilikum',
    'TomatoSauceOroDiParma','VollMilch','WeideMilchSmall','WhiteCeramicIkeaBowl','YellowCeramicPlate']
#############################################################
## get the gt and pred lists in one list
#############################################################
groundTruthList = []
predictionList = []
for i in range(0, 10):

  dirName = 'run' + str(i)
  predFileName = dirName + '/resultPred.p'
  gtFileName = dirName + '/resultGT.p'

  runGTList = pickle.load(open(gtFileName, 'r'))
  runPredList = pickle.load(open(predFileName, 'r'))
  for e in runGTList:
    groundTruthList.append(e)
  for e in runPredList:
    predictionList.append(e)

#############################################################
## change list to only contain the objectnames
#############################################################
## create results Directory
dirName = 'results'
if os.path.isdir(dirName):
    shutil.rmtree(dirName)
os.mkdir(dirName)

## failsafe copy
gtListFile = dirName + '/groundTruthListRaw.p'
pickle.dump(groundTruthList, open(gtListFile, 'w'))
predListFile = dirName + '/predictionListRaw.p'
pickle.dump(predictionList, open(predListFile, 'w'))

## removes clutter from classification labels
for x, entry in enumerate(groundTruthList):
  groundTruthList[x] = entry.split(',')[1].split(')')[0]

for x, entry in enumerate(predictionList):
  predictionList[x] = entry.split(',')[1].split(')')[0]

## failsafe copy
gtListFile = dirName + '/groundTruthList.p'
pickle.dump(groundTruthList, open(gtListFile, 'w'))
predListFile = dirName + '/predictionList.p'
pickle.dump(predictionList, open(predListFile, 'w'))

##########################################################
## metrics
##########################################################
fileName = dirName + '/metrics.txt'

fs = open(fileName, 'w')

#print 'accuracy: ' + str(accuracy_score(groundTruthList, predictionList))
fs.write('accuracy: ' + str(accuracy_score(groundTruthList, predictionList)) + '\n')

# #print 'precision(macro): ' + str(precision_score(groundTruthList, predictionList, average='macro'))
# #print 'precision(micro): ' + str(precision_score(groundTruthList, predictionList, average='micro'))
# #print 'precision(None): ' + str(precision_score(groundTruthList, predictionList, average=None))
fs.write('precision(macro): ' + str(round(precision_score(groundTruthList, predictionList, average='macro'), 4)) + '\n')
fs.write('precision(micro): ' + str(round(precision_score(groundTruthList, predictionList, average='micro'), 4)) + '\n')
fs.write('precision(None): ' + str(precision_score(groundTruthList, predictionList, average=None)) + '\n')

# #print 'recall(macro): ' + str(recall_score(groundTruthList, predictionList, average='macro'))
# #print 'recall(micro): ' + str(recall_score(groundTruthList, predictionList, average='micro'))
# #print 'recall(None): ' + str(recall_score(groundTruthList, predictionList, average=None))
fs.write('recall(macro): ' + str(round(recall_score(groundTruthList, predictionList, average='macro'), 4)) + '\n')
fs.write('recall(micro): ' + str(round(recall_score(groundTruthList, predictionList, average='micro'), 4)) + '\n')
fs.write('recall(None): ' + str(recall_score(groundTruthList, predictionList, average=None)) + '\n')

# #print 'f1(macro): ' + str(f1_score(groundTruthList, predictionList, average='macro'))
# #print 'f1(micro): ' + str(f1_score(groundTruthList, predictionList, average='micro'))
# #print 'f1(None): ' + str(f1_score(groundTruthList, predictionList, average=None))
fs.write('f1(macro): ' + str(round(f1_score(groundTruthList, predictionList, average='macro'), 4)) + '\n')
fs.write('f1(micro): ' + str(round(f1_score(groundTruthList, predictionList, average='micro'), 4)) + '\n')
fs.write('f1(None): ' + str(f1_score(groundTruthList, predictionList, average=None)) + '\n')

##################################################
## computes the accuracy for each class
################################################## 
cm = confusion_matrix(groundTruthList, predictionList, labels=objectList)
allCases = len(groundTruthList)
allCorrect = np.trace(cm)

accuracys = []
## for each row/label of the confusionMatrix compute truepositives, truenegatives, falsepositives and falsenegatives. 
for i in range(0, len(cm)):
    li = cm[i]
    tp = li[i]
    tn = allCorrect - tp
    rowAll = 0
    for k in range(0, len(li)):
        rowAll += li[k]
    
    fn = rowAll - tp
    fp = 0
    for k in range(0, len(cm)):
        fp += cm[k][i]
    fp -= tp
    ## compute the accuracy
    result = round((tp + tn) / float(tp + tn + fn + fp), 4)
    accuracys.append(result)

## write results and latex table to file
fs.write('class accuracys: ' + str(accuracys).strip('[]'))
fs.write('\n')
fs.write('\n')
fs.write(toLatexTab(objectList, accuracys, precision_score(groundTruthList, predictionList, average=None), recall_score(groundTruthList, predictionList, average=None), f1_score(groundTruthList, predictionList, average=None), 4 ))
fs.write('\n')
fs.write(toLatexTab(objectList, accuracys, precision_score(groundTruthList, predictionList, average=None), recall_score(groundTruthList, predictionList, average=None), f1_score(groundTruthList, predictionList, average=None), 2 ))
plot_confusion_matrix(groundTruthList, predictionList, labels=objectList, x_tick_rotation=90, figsize=(24,21), title=' ', text_fontsize='large', cmap='Reds')
matrixFileName = dirName + '/confusionMatrix.png'
plt.savefig(matrixFileName)
#plt.show()
fs.close()