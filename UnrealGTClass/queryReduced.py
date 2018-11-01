#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys
import pickle
import os
import numpy as np
import threading
import pracmln
from pracmln import MLN
from pracmln import Database
from pracmln import MLNQuery
from pracmln import MLNLearn
from sklearn.model_selection import KFold
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
from scikitplot.metrics import plot_confusion_matrix


if (len(sys.argv) < 3) | (len(sys.argv) > 3):
    print 'Falsche Paramteranzahl!'
else:
    i = sys.argv[1]
    print 'Starting query for directory: ' + str(i)
    ######################################
    ## testing of mln
    ######################################
    dirName = 'run' + str(i)
    mlnFileName = dirName + '/learnedMLN.mln'

    mln = MLN(grammar = 'PRACGrammar', logic = 'FirstOrderLogic')
    with( open(mlnFileName, 'r')) as f:
        for line in f:
            mln << line


    testFileName = dirName + '/' + sys.argv[2] +'testDB.txt'
    testDB = Database.load(mln, testFileName)

    pIdx = 0
    gtList = np.load(dirName + '/' + sys.argv[2] + 'GTraw.npy')
    dbpredList = []
    dbgtList = []
    for db in testDB:
        try:
            result = MLNQuery(mln=mln, db=db, method='WCSPInference', multicore=True, queries='object', cw=True, verbose=True).run()
        except pracmln.mln.errors.NoConstraintsError:
            continue
        ## find best result
        thisDBObjList = gtList[pIdx]
        for entry in thisDBObjList:
          predObj = entry
          try: 
            objVal = result.results[entry]
          except KeyError:
            continue
          for k, v in result.results.iteritems():
            if k.find(entry.split(',')[0])  != -1:
              if v > objVal:
                predObj = k
          dbpredList.append(predObj)
          dbgtList.append(entry)
        pIdx += 1
        ## -----------------
    predFileName = dirName + '/' + sys.argv[2] +'resultPred.p'
    pickle.dump(dbpredList, open(predFileName, 'w'))
    gtFileName = dirName + '/' + sys.argv[2] +'resultGT.p'
    pickle.dump(dbgtList, open(gtFileName, 'w'))

    del dbpredList
    del dbgtList
    del testDB


