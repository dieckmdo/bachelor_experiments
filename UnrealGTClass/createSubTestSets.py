#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import pickle
import numpy as np

dbFileName = 'UnrealGTClassClustered.txt'


delimiter = '---'
trainFileName = '/testDB.txt'
colorFileName = '/colortestDB.txt'
shapeFileName = '/shapetestDB.txt'
sizeFileName = '/sizetestDB.txt'
gogglesFileName = '/gogglestestDB.txt'
instanceFileName = '/instancetestDB.txt'

for i in range(0, 10):
    dirName = 'run' + str(i)
    train = np.load(dirName + '/testSetDistribution.npy')

    fs = open(dbFileName, 'r')
    dbs = fs.read().split(delimiter)

    cf = open(dirName+colorFileName, 'w')
    shf = open(dirName+shapeFileName, 'w')
    sizf = open(dirName+sizeFileName, 'w')
    inf = open(dirName+instanceFileName, 'w')
    gf = open(dirName+gogglesFileName, 'w')

    colorGT = []
    shapeGT = []
    sizeGT = []
    inGT = []
    gogGT = []

    for x in train:

        entry = []
        colorString = ''
        shapeString = ''
        sizeString = ''
        gogglesString = ''
        instanceString = ''
        sceneString = ''

        lines = dbs[x].split('\n')

        for l in lines:
            if l.find('color') != -1:
                colorString += l + '\n'
            elif l.find('shape') != -1:
                shapeString += l + '\n'
            elif l.find('size') != -1:
                sizeString += l + '\n'
            elif l.find('goggles') != -1:
                gogglesString += l + '\n'
            elif l.find('instance') != -1:
                instanceString += l + '\n'   
            elif l.find('scene') != -1:
                sceneString += l + '\n'
            elif l.find('object') != -1:
                entry.append(l.strip())
            
        if colorString != '':
            #cf.write(sceneString)                    
            cf.write(colorString)  
            cf.write(delimiter + '\n')  
            colorGT.append(entry)

        if shapeString != '':
            #shf.write(sceneString)                    
            shf.write(shapeString)  
            shf.write(delimiter + '\n') 
            shapeGT.append(entry)

        if sizeString != '':
            #sizf.write(sceneString)                    
            sizf.write(sizeString)  
            sizf.write(delimiter + '\n') 
            sizeGT.append(entry)

        if instanceString != '':
            #inf.write(sceneString)                    
            inf.write(instanceString)  
            inf.write(delimiter + '\n') 
            inGT.append(entry)

        if gogglesString != '':
            #gf.write(sceneString)                    
            gf.write(gogglesString)  
            gf.write(delimiter + '\n') 
            gogGT.append(entry)

    cf.close()
    shf.close() 
    sizf.close()
    inf.close()
    gf.close()
    colorgtFileName = dirName + '/colorGTraw.npy'
    np.save(colorgtFileName, colorGT)
    shapegtFileName = dirName + '/shapeGTraw.npy'
    np.save(shapegtFileName, shapeGT)
    sizegtFileName = dirName + '/sizeGTraw.npy'
    np.save(sizegtFileName, sizeGT)
    intgtFileName = dirName + '/instanceGTraw.npy'
    np.save(intgtFileName, inGT)
    goggtFileName = dirName + '/gogglesGTraw.npy'
    np.save(goggtFileName, gogGT)
