#!/usr/bin/env python
# -*- coding: utf-8 -*-

import pickle
import os
import numpy as np
from pracmln import MLN
from pracmln import Database
from pracmln import MLNQuery
from pracmln import MLNLearn
from sklearn.model_selection import KFold

###########################################################
## MLN learning and querieing
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
mln << '0 shape(?c, +?sha) ^ color(?c, +?col) ^ size(?c, +?size) ^ instance(?c, +?inst) ^ object(?c, +?obj)'
mln << '0 goggles_Logo(?c, +?comp) ^ object(?c, +?obj)'
mln << '0 goggles_Text(?c, +?text) ^ object(?c, +?obj)'
mln << '0 goggles_Product(?c, +?prod) ^ object(?c, +?obj)'
mln << '0 scene(+?s) ^ object(?c, +?obj)'
##unique clusters
mln << '#unique{+?t1,+?t2}'
mln << '0 object(?c1, +?t1) ^ object(?c2, +?t2) ^ ?c1 =/= ?c2'

dbFileName = 'UnrealGTClassClustered.txt'

#allDB = Database.load(mln, '/home/dominik/python_ws/testDB.txt')

allDB = Database.load(mln, dbFileName)

k = 0
splits = 10
testArray = np.array(allDB, dtype=Database)
kf = KFold(n_splits=splits, shuffle=True)
splitedArray = kf.split(testArray)
del allDB
for train, test in splitedArray:
    dirName = 'run' + str(k)
    os.mkdir(dirName)
    np.save(dirName + '/trainSetDistribution.npy', train)
    np.save(dirName + '/testSetDistribution.npy', test)
    print("%s %s" % (train, test))

    gtList = []

    ## read the input db
    delimiter = '---'
    trainFileName = dirName + '/trainDB.txt'
    testFileName = dirName + '/testDB.txt'
    fs = open(dbFileName, 'r')
    dbs = fs.read().split(delimiter)

    ## create train und test DB files based on the kfold splits
    ft = open(trainFileName, 'w')
    for x in train:
      ft.write(dbs[x])
      ft.write(delimiter)
    ft.close()
    del train

    ft = open(testFileName, 'w')
    for x in test:
      entry = []
      lines = dbs[x].split('\n')
      for l in lines:
        if l.find('object') != -1:
          entry.append(l.strip())

          ## if object line should occur in testfile
          #ft.write(l)
          #ft.write('\n')
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
