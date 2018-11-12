#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys
import pickle
import os
import numpy as np
import pracmln
from pracmln import MLN
from pracmln import Database
from pracmln import MLNQuery



if (len(sys.argv) < 3) | (len(sys.argv) > 3):
    print 'Falsche Paramteranzahl!'
else:
    i = sys.argv[1]
    dirName = 'run' + str(i) + '/' + sys.argv[2]
    print 'Starting query for directory: ' + dirName
    ######################################
    ## testing of mln
    ######################################
    mlnFileName = dirName + '/learnedMLN.mln'

    mln = MLN(grammar = 'PRACGrammar', logic = 'FirstOrderLogic')
    with( open(mlnFileName, 'r')) as f:
        for line in f:
            mln << line


    testFileName = dirName + '/testDB.txt'
    testDB = Database.load(mln, testFileName)

    pIdx = 0
    gtList = np.load(dirName + '/GTraw.npy')
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
    predFileName = dirName + '/resultPred.p'
    pickle.dump(dbpredList, open(predFileName, 'w'))
    gtFileName = dirName + '/resultGT.p'
    pickle.dump(dbgtList, open(gtFileName, 'w'))

    del dbpredList
    del dbgtList
    del testDB


