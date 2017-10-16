#!/usr/bin/python

""" 
    This is the code to accompany the Lesson 2 (SVM) mini-project.

    Use a SVM to identify emails from the Enron corpus by their authors:    
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

from sklearn import svm
from sklearn.metrics import accuracy_score

### features_train = features_train[:len(features_train)/100] 
### labels_train = labels_train[:len(labels_train)/100] 

clf = svm.SVC(kernel= "rbf", C= 10000)
t0 = time()
clf.fit(features_train, labels_train)
print "training time:", round(time()-t0, 3), "s"

### Training time kernel = rbf, C =10500: 191.369 s
### Training time kernel = rbf, C =1000: 337.981s
### Training time features_train = features_train[:len(features_train)/100] 
### labels_train = labels_train[:len(labels_train)/100]:0.187s
### Taining time kernel = rbf , C = 1: 0.172s
### Training time kernel = rbf: 0.172s
### Training time kernel = rbf, C =100:796.839 s

t1 = time()
pred = clf.predict(features_test)
print "predicting time:", round(time()-t1, 3), "s"

### Predicitng time kernel= rbf , C= 10500: 18.216s
### Predicitng time kernel= rbf , C= 1000: 37.157s
### Predicting time for features_train = features_train[:len(features_train)/100] 
### labels_train = labels_train[:len(labels_train)/100] : 1.876s
### Prediciting time kernel= rbf, C=1: 2.0s
### Prediciting time kernel= rbf: 1.857s
### Prediciting time kernel= rbf, C= 100:83.188 s

print accuracy_score(pred, labels_test)

### Acurracy_score kernel = rbf, C =10500: 0.990898748578
### Acurracy_score kernel = rbf, C =1000: 0.982935153584
### Acurracy_score for features_train = features_train[:len(features_train)/100] 
### labels_train = labels_train[:len(labels_train)/100]: 0.821387940842
### Accuracy score kernel = rbf, C= 1: 0.616040955631
### Accuracy score kernel = rbf : 0.616040955631
### Accuracy score kernel = rbf, C = 100: 0.955062571104

print "10th: %r, 26th: %r, 50th: %r" % (pred[10], pred[26], pred[50])

# There are over 1700 test events, how many are predicted to be in the "Chris" (1) class?
# No. of predicted to be in the 'Chris'(1): 877 for (Kernel = rbf, C = 10000)

print "No. of predicted to be in the 'Chris'(1): %r" % sum(pred)


chris = []
for i in pred:
    if i == 1:
        chris.append(i)
### chris email C=100: 922       
### chris email: 877
### chris email for C = 1000: 879
### chris email for features_train = features_train[:len(features_train)/100] 
### labels_train = labels_train[:len(labels_train)/100]: 11774
        
print len(chris)
