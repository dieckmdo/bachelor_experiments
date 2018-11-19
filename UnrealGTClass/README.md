# UnrealGTClass

The folder contains the results of the 10-fold crossvalidation on Unreal-Imgages with objectclasses as groundtruth.
Also contains the results for the crossvalidations with only one predicate.

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
- createSubset.py: creates the train/test databases for a single predicate
- learnSubset.py: trains the mlns with a single predicate
- querySubset.py: queries the single predicate databases
- computeResultsSubset.py: computes the results for the a single predicate

## Workflow

1. clustering.py (if wanted)
2. setup.py
3. learn.py
4. query.py
5. computeResults.py

If you want to learn with only a single predicate:

6. createSubset.py
7. learnSubset.py
8. querySubset.py
9. computeesultsSubset.py

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

### computeResults.py

Creates the confusion matrix and other metrics for the classification. 

usage: `python computeResults.py`

Creates a folder called results, where the confusion matrix, metrices and some backup files of the groundtruth and prediction lists get stored. 

### createSubset.py

Creates the train and test sets for the 10-fold crossvalidation with only one predicate and the *scene* prediacte, based on the testSetDistribution.npy for each folder [0-9]. Databases will only include either [color, size, shape, instance, goggles] annotations. The train and test DB get saved in run[0-9]/predicateName. Also saves the groundtruth for the test sets in a seperate file.

usage: `python createSubset.py [color, size, shape, instance, goggles]`

Takes the predicate name, for which the train/test dbs should be created, as parameter. UnrealGTClassClustered.txt is the input Database file. Outputs [color, size, shape, instance, goggles] folder in run[0-9], trainDB.txt, testDB.txt, GTraw.npy.

### learnSubset.py
Trains mlns with only the specified predicate with the respective training set in run[0-9].

usage: `python learnSubset.py [color, size, shape, instance, goggles]`

Takes the predicate for which to train a mln as paramter. Outputs a learned.mln file in run[0-9]/[color, size, shape, instance, goggles].

### querySubset.py
Queries the trained MLN in each predicate folder with the testset and asks for the object. The best found solution for each cluster is then stored as the predicted class. Also stores the groundtruth for each cluster.

usage: `python querySubset.py [0-9] [color, size, shape, instance, goggles]` 

Takes the folder number and predicate name as parameter. Outputs a resultPred.p for the predicted class and resultGT.p for the groundtruth in the respective predicate folder.

### computeResultsSubset.py
Creates the confusion matrix and other metrics for the classification with only one predicate.

usage: `python computeResultsSubset.py [color, size, shape, instance, goggles]`

Creates a folder with the predicate name in the results folder, where the confusion matrix, metrices and some backup files of the groundtruth and prediction lists get stored. 