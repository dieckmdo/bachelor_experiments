#!/usr/bin/env python
# -*- coding: utf-8 -*-

import pickle
import os
import random
import shutil
import numpy as np

UnrealFileName = 'UnrealGTInstanceClustered.txt'
RealFileName = 'RealGTInstanceClustered.txt'

dirName = 'mlnData'
if os.path.isdir(dirName):
    shutil.rmtree(dirName)
os.mkdir(dirName)

delimiter = '---'
trainFileName = dirName + '/trainDB.txt'
testFileName = dirName + '/testDB.txt'

unrealFile = open(UnrealFileName, 'r')
realFile = open(RealFileName, 'r')
trainFile = open(trainFileName, 'w')

trainFile.write(unrealFile.read())
unrealFile.close()

## 1/3 of the real data as train 
dbs = realFile.read().split(delimiter)
realFile.close()
random.shuffle(dbs)
trainDBs = dbs[:(len(dbs)/3)]
testDBs = dbs[(len(dbs)/3):]

for db in trainDBs: 
    trainFile.write(db)
    trainFile.write(delimiter)
    

trainFile.close()

## this will hold all groundtruth
gtList = []

## write the testfile with the remaining real dbs
testFile = open(testFileName, 'w')
for x in testDBs:
  # this will hold the groundtruth for that db
  entry = []
  lines = x.split('\n')
  for l in lines:
    ## omit object predicate / groundtruth
    if l.find('object') != -1:
      entry.append(l.strip())
    else:
      testFile.write(l)
      testFile.write('\n')
  testFile.write(delimiter)
  gtList.append(entry)
testFile.close()
del dbs

gtFileName = dirName + '/GTraw.npy'
np.save(gtFileName, gtList)
