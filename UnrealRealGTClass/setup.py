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

dbFileName = 'RealGTClassClustered.txt'

dirName = 'mlnData'
gtList = []

## read the input db
delimiter = '---'
testFileName = dirName + '/testDB.txt'
fs = open(dbFileName, 'r')
dbs = fs.read().split(delimiter)

ft = open(testFileName, 'w')
for x in dbs:
  entry = []
  lines = x.split('\n')
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
fs.close()
del dbs

gtFileName = dirName + '/GTraw.npy'
np.save(gtFileName, gtList)
