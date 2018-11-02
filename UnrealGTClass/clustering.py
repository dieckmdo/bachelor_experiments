#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import numpy as np
import sklearn.cluster
import distance

dbFile = 'UnrealGTClass.txt'
outFile = 'UnrealGTClassClustered.txt'
textWords = []
productWords = []
logoWords = []

fs = open(dbFile, 'r')

if os.path.isfile(outFile):
    os.remove(outFile)

fo = open(outFile, 'w')

## get the strings from the predicates
lines = fs.readlines()
for line in lines:
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

## clustering of text labels
text_lev_similarity = -1*np.array([[distance.levenshtein(w1,w2) for w1 in textWords] for w2 in textWords])

affprop = sklearn.cluster.AffinityPropagation(affinity="precomputed", damping=0.7)
affprop.fit(text_lev_similarity)
for cluster_id in np.unique(affprop.labels_):
    exemplar = textWords[affprop.cluster_centers_indices_[cluster_id]]
    cluster = np.unique(textWords[np.nonzero(affprop.labels_==cluster_id)])
    cluster_str = ", ".join(cluster)
    print(" - *%s:* %s" % (exemplar, cluster_str))

    ## replace the lines with the center cluster
    for i in range(0, len(lines)):
        newLine = lines[i]
        if newLine.find('goggles_Text') != -1:

            toreplace = newLine.split(',')[1].split(')')[0].strip()
            
            if toreplace in cluster:
                newLine = newLine.replace(toreplace, exemplar)
            lines[i] = newLine 


## clustering of product labels
product_lev_similarity = -1*np.array([[distance.levenshtein(w1,w2) for w1 in productWords] for w2 in productWords])

affprop = sklearn.cluster.AffinityPropagation(affinity="precomputed", damping=0.7)
affprop.fit(product_lev_similarity)
for cluster_id in np.unique(affprop.labels_):
    exemplar = productWords[affprop.cluster_centers_indices_[cluster_id]]
    cluster = np.unique(productWords[np.nonzero(affprop.labels_==cluster_id)])
    cluster_str = ", ".join(cluster)
    print(" - *%s:* %s" % (exemplar, cluster_str))

    ## replace the lines with the center cluster
    for i in range(0, len(lines)):
        newLine = lines[i]
        if newLine.find('goggles_Product') != -1:

            toreplace = newLine.split(',')[1].split(')')[0].strip()
            
            if toreplace in cluster:
                newLine = newLine.replace(toreplace, exemplar)
            lines[i] = newLine 


## clustering of logo labels
logo_lev_similarity = -1*np.array([[distance.levenshtein(w1,w2) for w1 in logoWords] for w2 in logoWords])

affprop = sklearn.cluster.AffinityPropagation(affinity="precomputed", damping=0.7)
affprop.fit(logo_lev_similarity)
for cluster_id in np.unique(affprop.labels_):
    exemplar = logoWords[affprop.cluster_centers_indices_[cluster_id]]
    cluster = np.unique(logoWords[np.nonzero(affprop.labels_==cluster_id)])
    cluster_str = ", ".join(cluster)
    print(" - *%s:* %s" % (exemplar, cluster_str))

    ## replace the lines with the center cluster
    for i in range(0, len(lines)):
        newLine = lines[i]
        if newLine.find('goggles_Logo') != -1:

            toreplace = newLine.split(',')[1].split(')')[0].strip()
            
            if toreplace in cluster:
                newLine = newLine.replace(toreplace, exemplar)
            lines[i] = newLine 

fo.writelines(lines)
fs.close()
fo.close()