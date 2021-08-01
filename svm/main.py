"""
    This is the code to accompany the Lesson 2 (SVM) mini-project.
    Use a SVM to identify emails from the Enron corpus by their authors:
    Sara has label 0
    Chris has label 1
"""

import sys
from time import time

from email_preprocess import preprocess
from sklearn import svm

### features_train and features_test are the features for the training
### and testing datasets, respectively
### labels_train and labels_test are the corresponding item labels
features_train, features_test, labels_train, labels_test = preprocess()

#########################################################
### your code goes here ###

clf = svm.SVC( kernel ='rbf',C=10000)
t0 = time()
clf.fit(features_train, labels_train)
t1 = time()
print(t1 - t0)
predict= clf.predict(features_test)
t2 = time()
print(t2 - t1)
print(clf.score(features_test, labels_test))
#########################################################
