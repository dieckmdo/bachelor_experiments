# ClassifierGTClass

This folder contains the results from the class classification of the `RFAnnotator` and `SVMAnnotator` from the robosherlock package rs_addons.
For the trained models see ...

The Annotators added their classification as *instance* predicate. The groundtruth was annotated using the `UnrealGTAnnotator` and is represented by the *object* predicate.

## Contents

- RFresults: contains the confusion matrix and metrics for the classification of the `RFAnnotator`
- SVMresults: contains the confusion matrix and metrics for the classification of the `SVMAnnotator`
- RFUnrealClassifier.txt: the merged database file of the `RFAnnotator`
- SVMUnrealClassifier.txt: the merged database file of the `SVMAnnotator`
- validation.py: a script to compute the confusion matrix and metrics for the classifications

- the database files for the scenes splitted in scenarios, because the scenes get run through robosherlock per scenario

## validation.py

usage: `python validation.py [RF, SVM]` 

Specifies which database file should be used and the name of the result folder. Warning: If the results folder already exists, it gets deleted and created anew.
Database files should always be of the form *classifier*UnrealClassifier.txt for the script to work.

