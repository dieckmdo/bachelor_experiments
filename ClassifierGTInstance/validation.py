#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import sys
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

def computeResults(classifier):
    
    dbFileName = classifier + 'ClassifierInstance.txt'

    predictionList = []
    groundTruthList = []

    ## get the classification and groudtruth from the corresponding database file
    delimiter = '---'
    fs = open(dbFileName, 'r')
    dbs = fs.read().split(delimiter)

    for db in dbs:
        lines = db.split('\n')
        for l in lines:
            if l.find('object') > -1:
                groundTruthList.append(l)
            elif l.find('instance') > -1:
                predictionList.append(l)

    #############################################################
    ## change list to only contain the objectnames
    #############################################################
    ## create results Directory
    dirName = classifier + 'results'
    try:
        os.mkdir(dirName)
    except OSError:
        shutil.rmtree(dirName)
        os.mkdir(dirName)

    for x, entry in enumerate(groundTruthList):
      groundTruthList[x] = entry.split(',')[1].split(')')[0]

    for x, entry in enumerate(predictionList):
      predictionList[x] = entry.split(',')[1].split(')')[0]

    ##########################################################
    ## metrics
    ##########################################################
    objectList = ['AlbiHimbeerJuice','BlueCeramicIkeaMug','BlueMetalPlateWhiteSpeckles','BluePlasticBowl','BluePlasticFork','BluePlasticKnife','BluePlasticSpoon','CupEcoOrange','EdekaRedBowl','ElBrygCoffee','JaMilch','JodSalz','KelloggsCornFlakes',
    'KelloggsToppasMini','KnusperSchokoKeks','KoellnMuesliKnusperHonigNuss','LargeGreySpoon','LinuxCup','LionCerealBox','MarkenSalz','MeerSalz','MondaminPancakeMix','NesquikCereal','PfannerGruneIcetea','PfannerPfirsichIcetea','RedMetalBowlWhiteSpeckles',
    'RedMetalCupWhiteSpeckles','RedMetalPlateWhiteSpeckles','RedPlasticFork','RedPlasticKnife','RedPlasticSpoon','ReineButterMilch','SeverinPancakeMaker','SiggBottle','SlottedSpatula','SojaMilch','SpitzenReis','TomatoAlGustoBasilikum',
    'TomatoSauceOroDiParma','VollMilch','WeideMilchSmall','WhiteCeramicIkeaBowl','YellowCeramicPlate']
    fileName = dirName + '/metrics.txt'

    fs = open(fileName, 'w')

    #print 'accuracy: ' + str(accuracy_score(groundTruthList, predictionList))
    fs.write('accuracy: ' + str(accuracy_score(groundTruthList, predictionList)) + '\n')

    # #print 'precision(macro): ' + str(precision_score(groundTruthList, predictionList, average='macro'))
    # #print 'precision(micro): ' + str(precision_score(groundTruthList, predictionList, average='micro'))
    # #print 'precision(None): ' + str(precision_score(groundTruthList, predictionList, average=None))
    fs.write('precision(macro): ' + str(round(precision_score(groundTruthList, predictionList, average='macro'), 4)) + '\n')
    fs.write('precision(micro): ' + str(round(precision_score(groundTruthList, predictionList, average='micro'), 4)) + '\n')
    fs.write('precision(weighted): ' + str(round(precision_score(groundTruthList, predictionList, average='weighted'), 4)) + '\n')
    fs.write('precision(None): ' + str(precision_score(groundTruthList, predictionList, average=None)) + '\n')

    # #print 'recall(macro): ' + str(recall_score(groundTruthList, predictionList, average='macro'))
    # #print 'recall(micro): ' + str(recall_score(groundTruthList, predictionList, average='micro'))
    # #print 'recall(None): ' + str(recall_score(groundTruthList, predictionList, average=None))
    fs.write('recall(macro): ' + str(round(recall_score(groundTruthList, predictionList, average='macro'), 4)) + '\n')
    fs.write('recall(micro): ' + str(round(recall_score(groundTruthList, predictionList, average='micro'), 4)) + '\n')
    fs.write('recall(weighted): ' + str(round(recall_score(groundTruthList, predictionList, average='weighted'), 4)) + '\n')
    fs.write('recall(None): ' + str(recall_score(groundTruthList, predictionList, average=None)) + '\n')

    # #print 'f1(macro): ' + str(f1_score(groundTruthList, predictionList, average='macro'))
    # #print 'f1(micro): ' + str(f1_score(groundTruthList, predictionList, average='micro'))
    # #print 'f1(None): ' + str(f1_score(groundTruthList, predictionList, average=None))
    fs.write('f1(macro): ' + str(round(f1_score(groundTruthList, predictionList, average='macro'), 4)) + '\n')
    fs.write('f1(micro): ' + str(round(f1_score(groundTruthList, predictionList, average='micro'), 4)) + '\n')
    fs.write('f1(weighted): ' + str(round(f1_score(groundTruthList, predictionList, average='weighted'), 4)) + '\n')
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
        line = cm[i]
        tp = line[i]
        tn = allCorrect - tp
        rowAll = 0
        for k in range(0, len(line)):
            rowAll += line[k]
        
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

def main():
    if (len(sys.argv) != 2):
        print 'Wrong paramter count! Needs a parameter for the classifier name.'
    else:
        computeResults(sys.argv[1])

if __name__=="__main__":
    main()
    