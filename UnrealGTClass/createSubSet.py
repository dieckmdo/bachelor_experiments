#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import sys
import pickle
import shutil
import numpy as np


def createSet(pred):

    dbFileName = 'UnrealGTClassClustered.txt'

    delimiter = '---'

    for i in range(0, 10):
        dirNameBase = 'run' + str(i)
        

        dirNamePred = dirNameBase + '/' + pred
        if os.path.isdir(dirNamePred):
            shutil.rmtree(dirNamePred)
        os.mkdir(dirNamePred)

        fs = open(dbFileName, 'r')
        dbs = fs.read().split(delimiter)

        ## trainset
        train = np.load(dirNameBase + '/trainSetDistribution.npy')
        trainFile = open(dirNamePred + '/trainDB.txt', 'w')

        for x in train:

            predString = ''
            sceneString = ''
            objectString = ''

            lines = dbs[x].split('\n')

            ## look for the predicate names
            for l in lines:
                if l.find(pred) != -1:
                    predString += l + '\n'
                elif l.find('scene') != -1:
                    sceneString += l + '\n'
                elif l.find('object') != -1:
                    objectString += l +'\n'
                
            ## if a predicate occured at least ones in a db, write it to the testDB file
            ## the respective groundtruth gets saved too
            if predString != '':
                trainFile.write(sceneString)                    
                trainFile.write(predString)  
                trainFile.write(objectString)
                trainFile.write(delimiter + '\n')  

        trainFile.close()

        ## testset
        test = np.load(dirNameBase + '/testSetDistribution.npy')
        testFile = open(dirNamePred + '/testDB.txt', 'w')
        gtList = []

        for x in test:

            entry = []
            predString = ''
            sceneString = ''

            lines = dbs[x].split('\n')

            ## look for the predicate names
            for l in lines:
                if l.find(pred) != -1:
                    predString += l + '\n'
                elif l.find('scene') != -1:
                    sceneString += l + '\n'
                elif l.find('object') != -1:
                    entry.append(l.strip())
                
            ## if a predicate occured at least ones in a db, write it to the testDB file
            ## the respective groundtruth gets saved too
            if predString != '':
                testFile.write(sceneString)                    
                testFile.write(predString)  
                testFile.write(delimiter + '\n')  
                gtList.append(entry)

        testFile.close()
        fs.close()

        gtFileName = dirNamePred + '/GTraw.npy'
        np.save(gtFileName, gtList)


def main():
    if len(sys.argv) != 2:
        print 'Wrong parameter count. Specify which prediate to use. Example color.'
    else:
        createSet(sys.argv[1])

if __name__=="__main__":
    main()
