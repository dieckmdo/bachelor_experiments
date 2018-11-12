#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import pracmln
from pracmln import MLN
from pracmln import Database
from pracmln import MLNLearn

def runMLN():
    ###########################################################
    ## MLN instanciating
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

    dirName = 'mlnData'
    trainFileName = dirName + '/trainDB.txt'
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
    runMLN()

if __name__=="__main__":
    main()
