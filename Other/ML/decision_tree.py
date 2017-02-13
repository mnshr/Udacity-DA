# -*- coding: utf-8 -*-
"""
Created on Sun Jan 22 22:05:45 2017

@author: Manish
"""

import sys
from class_vis import prettyPicture
from prep_terrain_data import makeTerrainData

import numpy as np
import pylab as pl
from sklearn import tree
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

features_train, labels_train, features_test, labels_test = makeTerrainData()



#################################################################################


########################## DECISION TREE #################################



#### your code goes here
clf = tree.DecisionTreeClassifier(min_samples_split=50)
clf = clf.fit(features_train, labels_train)
pred=clf.predict(features_test)

acc = accuracy_score(pred, labels_test)
### be sure to compute the accuracy on the test set
print('accuracy is: ', acc)


def submitAccuracies():
  return {"acc":round(acc,3)}

