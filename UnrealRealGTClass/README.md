# UnrealGTClass

The folder contains the results of the classification of the real images from and MLN, that was trained on Unreal-Images.

## Contents

- results: contains the results of classification with the confusion matrix and metrics
- UnrealGTClass.txt: database file of all Unreal-Images
- RealGTClass.txt: database file of all real images
- UnrealGTClassClustered.txt: the database file, but with clustered *goggles* strings
- RealGTClassClustered.txt: the database file, but with clustered *goggles* strings
- clustering.py: clusters the goggles strings
- setup.py: sets up the real database file for testing it on a mln
- learn.py: trains the mln
- query.py: queries the mln
- computeResults.py: computes the results

## Workflow

1. clustering.py (if wanted)
2. setup.py
3. learn.py
4. query.py
5. computeResults.py

What the each script does and how it is called is described below. 

### clustering.py

Clusters the *goggles* strings per predicates and replaces the strings with their cluster centroid. So the results of *goggles_Text*, *goggles_Logo* and *goggles_Product* get clustered independently.
Uses affinity propagation and the levenshtein distance.

usage: `python clustering.py`

Takes the UnrealGTClass.txt and the RealGTClass.txt as input and outputs the database with clustered strings as UnrealGTClassClustered.txt and RealGTClassClustered.txt.

### setup.py

Creates the test database file from the real database file, by ommiting the groundtruth, as it should be classified on this later. The gt gets stored in a seperate file.

usage: `python setup.py`

Takes the RealGTClassClustered.txt as input as Database file. Outputs the folder mlnData were testDB.txt and GTraw.npy gets stored.

### learn.py

Train the mln.

usage: `python learn.py`

Outputs the learnedMLN.mln file in the mlnData folder.

### query.py

Queries the trained MLN with the real images and asks for the object. The best found solution for each cluster is then stored as the predicted class. Also stores the groundtruth for each cluster.

usage: `python query.py`

Outputs a resultPred.p for the predicted class and resultGT.p for the groundtruth in mlnData.

### computeResults

Creates the confusion matrix and other metrics for the classification. 

usage: `python computeResults.py`

Creates a folder called results, where the confusion matrix, metrices and some backup files of the groundtruth and prediction lists get stored. 