Project: Final Project: CSC522 Automated learning and Data Analysis
Authors: Nathan Climer <njclimer@ncsu.edu>,
         Charles Haithcock <cehaith2@ncsu.edu>
Prerequisites: Python 2.0, Numpy, Joblib, sklearn, numpy, scipy, ntlk, pdb,
               matplotlib

===============================FILE CONTENTS===================================
-Overview
-Usage
===================================OVERVIEW====================================

This codebase evaluates several classifiers on Stack Overflow questions.
Classifiers included are:
 - baysean classifier with word frequency priors.
 - baysean classifier with document presense prios.
 - K-Nearest Neighbors
 - One-vs-All Support Vector Machine
 - A binary relevance wrapper for any supplied binary classifier.
=====================================USAGE=====================================
All scripts are located in the bin folder.

All code is captured in python scripts that will need to be modified to run in
your environment. First, the baysean classifier code consists of 4 files which
all have the same layout.
 - baysean_classifier.py (tests word frequency priors on PT2 selected data)
 - baysean_classifier_binary_relevance.py (Word frequency with binary relevance)
 - baysean_classifier_doc.py (tests document presence on PT2 selected data)
 - baysean_classifier_binary_relevance_doc.py (Document presense prior with biary relevance)

To use these files, change the PROJECT_PATH to the source of your project. You may
also want to select a different or smaller subset of tags. Then simply call
python -O baysean_classifier.py

For the KNN and OVA SVM classifiers, if the current working directory is autotagger/, 
simply run
 python [-i] bin/knn_svm.py data/plain-text/full.csv
for KNN, or
 python [-i] bin/ova_svm.py data/plain-text/full.csv
for OVA SVM.

Uncomment line 117 to also run kNN.

Other files of interest:
eval_classifier.py contains methods for evaluating the classifiers.
naive_bayse.py contains methods for training and classifing using the
    baysean classifier.
tokens.py, preprocess_to_questions.py - preprocesses the data.
svm.py - training and classifying KNN and OVA SVM.