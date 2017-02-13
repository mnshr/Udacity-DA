# -*- coding: utf-8 -*-
"""
Created on Tue Jan 17 06:41:47 2017

@author: Manish
"""

import sys
sys.path.append("../tools/ud120-projects-master/choose_your_own")
from class_vis import prettyPicture
from prep_terrain_data import makeTerrainData

import matplotlib.pyplot as plt
import copy
import numpy as np
import pylab as pl

features_train, labels_train, features_test, labels_test = makeTerrainData()


########################## SVM #################################
### we handle the import statement and SVC creation for you here
from sklearn.svm import SVC
clf = SVC(kernel="rbf", gamma=10, C=1000000)

clf.fit(features_train, labels_train)

pred=clf.predict(features_test)
#Plot it
prettyPicture(clf, features_test, labels_test)

from sklearn.metrics import accuracy_score
acc = accuracy_score(pred, labels_test)
print ("the accuracy is: ", acc)

def submitAccuracy():
    return acc