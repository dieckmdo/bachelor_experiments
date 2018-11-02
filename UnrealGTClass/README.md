# UnrealGTClass

The folder contains the results of the 10-fold crossvalidation on Unreal-Imgages with objectclasses as groundtruth.

## Contents

- results: contains the results of the 10-fold crossvalidation
    - also contains folders for the crossvalidation on single Annotator evidence
- run[0-9]: each folder contains one learned MLN for one run of the 10-fold crossvalidation
- UnrealGTClass.txt: database file of all Unreal-Images
- UnrealGTClassClustered.txt: the database file, but with clustered *goggles* strings
- clustering.py: clusters the goggles strings
- setup.py: creates and setups the directories and train/testsets for crossvalidation
- learn.py: trains the mlns
- query.py: queries the mlns
- computeResults.py: computes the results
- createSubTestSets.py
- queryReduced.py
- computeResultsReduced.py

## Workflow

1. clustering.py (if wanted)
2. setup.py
3. learn.py
4. query.py

What the each script does and how it is called is described below. 

### clustering.py

Clusters the *goggles* strings per predicates and replaces the strings with their cluster centroid. So the results of *goggles_Text*, *goggles_Logo* and *goggles_Product* get clustered independently.
Uses affinity propagation and the levenshtein distance.

usage: `python clustering.py`

Takes the UnrealGTClass.txt as input and outputs the database with clusters strings as UnrealGTClassClustered.txt

### setup.py

Creates the train and test sets for the 10-fold crossvalidation. They get saved in their respective folder run[0-9]. Also creates the train and test db based on the distributions. 
The groundtruth for each test set get omitted while writing the db, as this should be classified later. The gt gets stored in a seperate file.

usage: `python setup.py`

Takes the UnrealGTClassClustered.txt as input as Database file. Outputs 10 folders with their respective files: [test,train]SetDistrubution.npy, trainDB.txt, testDB.txt, GTraw.npy.

### learn.py

Trains an mln for each run[0-9] folder with the respective trainingset.

usage: `python learn.py`

Outputs a learnedMLN.mln file for each folder.

Note: The learning takes place sequentially for every folder as the learning needs a lot of memory (this was done on a 16GB Ram machine). If you think you have enough memory to handle it, there is a threading solution commented out. 

### query.py

Queries the trained MLN in each folder with the testset and asks for the object. The best found solution for each cluster is then stored as the predicted class. Also stores the groundtruth for each cluster.

usage: `python query.py [0-9]`

Takes the number of the folder which should be queried as parameter. Outputs a resultPred.p for the predicted class and resultGT.p for the groundtruth.

### computeResults

Creates the confusion matrix and other metrics for the classification. 

usage: `python computeResults.py`

Creates a folder called results, where the confusion matrix, metrices and some backup files of the groundtruth and prediction lists get stored. 