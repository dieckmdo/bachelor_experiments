#!/usr/bin/env python
# -*- coding: utf-8 -*-

import pickle
import os
import shutil
import numpy as np
from pracmln import MLN
from pracmln import Database
from pracmln import MLNQuery
from pracmln import MLNLearn
from sklearn.model_selection import KFold

###########################################################
## instanciate MLN
###########################################################
mln = MLN(grammar = 'PRACGrammar', logic = 'FirstOrderLogic')
##predicates
mln << 'shape(cluster, shape)'
mln << 'color(cluster, color)'
mln << 'size(cluster, size)'
mln << 'goggles_Text(cluster, text)'
mln << 'goggles_Logo(cluster, company)'
mln << 'goggles_Product(cluster, product)'
mln << 'scene(scene)'
mln << 'instance(cluster, instance)'
mln << 'object(cluster, object!)'
##formulas
mln << '0 shape(?c, +?sha) ^ object(?c, +?obj)'
mln << '0 color(?c, +?col) ^ object(?c, +?obj)'
mln << '0 size(?c, +?size) ^ object(?c, +?obj)'
mln << '0 instance(?c, +?inst) ^ object(?c, +?obj)'
mln << '0 goggles_Logo(?c, +?comp) ^ object(?c, +?obj)'
mln << '0 goggles_Text(?c, +?text) ^ object(?c, +?obj)'
mln << '0 goggles_Product(?c, +?prod) ^ object(?c, +?obj)'
mln << '0 scene(+?s) ^ object(?c, +?obj)'
##unique clusters
mln << '#unique{+?t1,+?t2}'
mln << '0 object(?c1, +?t1) ^ object(?c2, +?t2) ^ ?c1 =/= ?c2'


dbFileName = 'UnrealGTClassClustered.txt'
allDB = Database.load(mln, dbFileName)

## create the train and test splits and the corresponding databases
k = 0
splits = 10
testArray = np.array(allDB, dtype=Database)
kf = KFold(n_splits=splits, shuffle=True)
splitedArray = kf.split(testArray)
del allDB
for train, test in splitedArray:
    dirName = 'run' + str(k)
    if os.path.isdir(dirName):
      shutil.rmtree(dirName)
    os.mkdir(dirName)
    np.save(dirName + '/trainSetDistribution.npy', train)
    np.save(dirName + '/testSetDistribution.npy', test)
    print("%s %s" % (train, test))

    ## this will hold all groundtruth for the corresponding testset
    gtList = []

    ## read the input db
    delimiter = '---'
    trainFileName = dirName + '/trainDB.txt'
    testFileName = dirName + '/testDB.txt'
    fs = open(dbFileName, 'r')
    dbs = fs.read().split(delimiter)

    ## create train file
    ft = open(trainFileName, 'w')
    for x in train:
      ft.write(dbs[x])
      ft.write(delimiter)
    ft.close()
    del train

    # create the test file
    ft = open(testFileName, 'w')
    for x in test:
      ## this will hold the groundtruth for that db
      entry = []
      lines = dbs[x].split('\n')
      for l in lines:
        ## omit object predicate / groundtruth
        if l.find('object') != -1:
          entry.append(l.strip())
        else:
          ft.write(l)
          ft.write('\n')
      ft.write(delimiter)
      gtList.append(entry)
    ft.close()
    del test
    fs.close()
    del dbs

    gtFileName = dirName + '/GTraw.npy'
    np.save(gtFileName, gtList)
    k += 1
