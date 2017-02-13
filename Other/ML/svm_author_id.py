#!/usr/bin/python

"""
    This is the code to accompany the Lesson 2 (SVM) mini-project.

    Use a SVM to identify emails from the Enron corpus by their authors:
    Sara has label 0
    Chris has label 1
"""

import sys
from time import time
sys.path.append("../tools/ud120-projects-master/tools")
sys.path.append("../tools/ud120-projects-master/choose_your_own")
from email_preprocess import preprocess
from operator import *

### features_train and features_test are the features for the training
### and testing datasets, respectively
### labels_train and labels_test are the corresponding item labels
features_train, features_test, labels_train, labels_test = preprocess()




#########################################################
### your code goes here ###

#########################################################


from sklearn.svm import SVC
import math as ml
clf = SVC(kernel="rbf", C=10000) #, gamma=1.0, C=1.0)

t0 = time()
#features_train = features_train[:math.floor(len(features_train)/100)]
#labels_train = labels_train[:math.floor(len(labels_train)/100)]
clf.fit(features_train, labels_train)
print ("training time:", round(time()-t0, 3), "s")

t0 = time()
pred=clf.predict(features_test)
print ("predict time:", round(time()-t0, 3), "s")

#Plot it
from class_vis import prettyPicture
#prettyPicture(clf, features_test, labels_test)

from sklearn.metrics import accuracy_score
acc = accuracy_score(pred, labels_test)
print ("the accuracy is: ", acc)
#print ("10th: ", pred[10])
#print ("26th: ", pred[26])
#print ("50th: ", pred[50])
j = 0
for i in pred:
    if i == 1:
        j=j+1
print ("Counts of 1:", j)

