#!/usr/bin/python

""" 
    This is the code to accompany the Lesson 3 (decision tree) mini-project.

    Use a Decision Tree to identify emails from the Enron corpus by author:    
    Sara has label 0
    Chris has label 1
"""
    
import sys
from time import time
sys.path.append("../tools/")
from email_preprocess import preprocess

### features_train and features_test are the features for the training
### and testing datasets, respectively
### labels_train and labels_test are the corresponding item labels
features_train, features_test, labels_train, labels_test = preprocess()




#########################################################
### your code goes here ###


#########################################################
from sklearn import tree
from sklearn.metrics import accuracy_score

clf = tree.DecisionTreeClassifier(min_samples_split = 40)

t0 = time()

clf.fit(features_train, labels_train)

print "training time:", round(time()-t0, 3), "s"

## Training time for min_samples_split = 40: 109.027 s
## Training time for percentile = 1 & min_samples_split = 40: 9.407s

t1 = time()
pred = clf.predict(features_test)
print "prediciting_time:", round(time()-t1, 3), "s"

## Prediciting time for min_samples_split = 40: 0.059 s
## Prediciting time fot percentile 1 & min_samples_split = 40 : 0.0s

print "Accuracy:", accuracy_score(pred, labels_test)

## Accuracy for min_samples_split = 40 : 0.976678043231
## Accuracy for percentile = 1 & min_samples_split = 40: 0.966439135381

## Number of features for the datasets: 3785
## Number of features when percentile = 1 in email_preprocess.py: 379

print "No. of features:", len(features_train[0])

## No of email = 869
## No of email in the case of percentile = 1: 887
chris  = []
for i in pred:
    if i == 1:
        chris.append(i)
print len(chris)
