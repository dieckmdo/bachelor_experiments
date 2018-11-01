#!/usr/bin/env python
# -*- coding: utf-8 -*-

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

def runMLN(k):
    print 'run' + str(k)
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

    dirName = 'run' + str(k)
    trainFileName = dirName + '/trainDB.txt'
    testFileName = dirName + '/testDB.txt'
    ###################################
    ## learning of mln
    #################################
    trainDB = Database.load(mln, trainFileName)

    learndMLN = MLNLearn(mln=mln, db=trainDB, method='DBPLL_CG', discr_preds=pracmln.QUERY_PREDS, qpreds='object', multicore=False, verbose=True, use_prior=True, prior_mean=0, prior_stdev=10, optimizer='',ignore_unknown_predicates=True).run()

    learndFileName = dirName + '/learnedMLN.mln'
    fs = open(learndFileName, 'w')
    learndMLN.write(stream=fs)
    fs.close()

    del trainDB

def main():
    for i in range(0, 10):
       runMLN(i)
    # multithread here
    #threads = []
    #for i in range(2, 10):
     #   t = threading.Thread(target=runMLN, args=(i,))
     #   threads.append(t)
     #   t.start()
     #   print "Start thread no. " + str(i)

    #for t in threads:
    #    t.join()

    print "All done"

if __name__=="__main__":
    main()
