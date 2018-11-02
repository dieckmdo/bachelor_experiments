#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import numpy as np
import sklearn.cluster
import distance

dbFileU = 'UnrealGTInstance.txt'
dbFileR = 'RealGTInstance.txt'
outUnreal = 'UnrealGTInstanceClustered.txt'
outReal = 'RealGTInstanceClustered.txt'

textWords = []
productWords = []
logoWords = []

fsU = open(dbFileU, 'r')
fsR = open(dbFileR, 'r')

if os.path.isfile(outUnreal):
    os.remove(outUnreal)
foU = open(outUnreal, 'w')

if os.path.isfile(outReal):
    os.remove(outReal)
foR = open(outReal, 'w')


linesU = fsU.readlines()
linesR = fsR.readlines()

## get the strings from the predicates
for line in linesU:
    if line.find('goggles') != -1:
        gogType = line.split('(')

        if gogType[0].find('Logo') != -1:
            logoWords.append(gogType[1].split(')')[0].split(',')[1].strip())
        elif gogType[0].find('Product') != -1:
            productWords.append(gogType[1].split(')')[0].split(',')[1].strip())
        elif gogType[0].find('Text') != -1:
            textWords.append(gogType[1].split(')')[0].split(',')[1].strip())

for line in linesR:
    if line.find('goggles') != -1:
        gogType = line.split('(')

        if gogType[0].find('Logo') != -1:
            logoWords.append(gogType[1].split(')')[0].split(',')[1].strip())
        elif gogType[0].find('Product') != -1:
            productWords.append(gogType[1].split(')')[0].split(',')[1].strip())
        elif gogType[0].find('Text') != -1:
            textWords.append(gogType[1].split(')')[0].split(',')[1].strip())


textWords = np.asarray(textWords)
productWords = np.asarray(productWords)
logoWords = np.asarray(logoWords)

### clustering of text labels
text_lev_similarity = -1*np.array([[distance.levenshtein(w1,w2) for w1 in textWords] for w2 in textWords])

affprop = sklearn.cluster.AffinityPropagation(affinity="precomputed", damping=0.8)
affprop.fit(text_lev_similarity)
for cluster_id in np.unique(affprop.labels_):
    exemplar = textWords[affprop.cluster_centers_indices_[cluster_id]]
    cluster = np.unique(textWords[np.nonzero(affprop.labels_==cluster_id)])
    cluster_str = ", ".join(cluster)
    print(" - *%s:* %s" % (exemplar, cluster_str))

    ## replace the lines with the center cluster
    for i in range(0, len(linesU)):
        newLine = linesU[i]
        if newLine.find('goggles_Text') != -1:

            toreplace = newLine.split(',')[1].split(')')[0].strip()
            
            if toreplace in cluster:
                newLine = newLine.replace(toreplace, exemplar)
            linesU[i] = newLine 

                ## replace the lines with the center cluster
    for i in range(0, len(linesR)):
        newLine = linesR[i]
        if newLine.find('goggles_Text') != -1:

            toreplace = newLine.split(',')[1].split(')')[0].strip()
            
            if toreplace in cluster:
                newLine = newLine.replace(toreplace, exemplar)
            linesR[i] = newLine 


# ### clustering of product labels
product_lev_similarity = -1*np.array([[distance.levenshtein(w1,w2) for w1 in productWords] for w2 in productWords])

affprop = sklearn.cluster.AffinityPropagation(affinity="precomputed", damping=0.8)
affprop.fit(product_lev_similarity)
for cluster_id in np.unique(affprop.labels_):
    exemplar = productWords[affprop.cluster_centers_indices_[cluster_id]]
    cluster = np.unique(productWords[np.nonzero(affprop.labels_==cluster_id)])
    cluster_str = ", ".join(cluster)
    print(" - *%s:* %s" % (exemplar, cluster_str))

    ## replace the lines with the center cluster
    for i in range(0, len(linesU)):
        newLine = linesU[i]
        if newLine.find('goggles_Product') != -1:

            toreplace = newLine.split(',')[1].split(')')[0].strip()
            
            if toreplace in cluster:
                newLine = newLine.replace(toreplace, exemplar)
            linesU[i] = newLine 

                ## replace the lines with the center cluster
    for i in range(0, len(linesR)):
        newLine = linesR[i]
        if newLine.find('goggles_Product') != -1:

            toreplace = newLine.split(',')[1].split(')')[0].strip()
            
            if toreplace in cluster:
                newLine = newLine.replace(toreplace, exemplar)
            linesR[i] = newLine 


# ### clustering of logo labels
logo_lev_similarity = -1*np.array([[distance.levenshtein(w1,w2) for w1 in logoWords] for w2 in logoWords])

affprop = sklearn.cluster.AffinityPropagation(affinity="precomputed", damping=0.8)
affprop.fit(logo_lev_similarity)
for cluster_id in np.unique(affprop.labels_):
    exemplar = logoWords[affprop.cluster_centers_indices_[cluster_id]]
    cluster = np.unique(logoWords[np.nonzero(affprop.labels_==cluster_id)])
    cluster_str = ", ".join(cluster)
    print(" - *%s:* %s" % (exemplar, cluster_str))

    ## replace the lines with the center cluster
    for i in range(0, len(linesU)):
        newLine = linesU[i]
        if newLine.find('goggles_Logo') != -1:

            toreplace = newLine.split(',')[1].split(')')[0].strip()
            
            if toreplace in cluster:
                newLine = newLine.replace(toreplace, exemplar)
            linesU[i] = newLine 

                ## replace the lines with the center cluster
    for i in range(0, len(linesR)):
        newLine = linesR[i]
        if newLine.find('goggles_Logo') != -1:

            toreplace = newLine.split(',')[1].split(')')[0].strip()
            
            if toreplace in cluster:
                newLine = newLine.replace(toreplace, exemplar)
            linesR[i] = newLine  

foU.writelines(linesU)
foU.close()
fsU.close()

foR.writelines(linesR)
foR.close()
fsR.close()