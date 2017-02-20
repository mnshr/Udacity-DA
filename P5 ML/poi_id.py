# -*- coding: utf-8 -*-
"""
Created on Mon Feb 20 07:49:51 2017

@author: Manish
"""

#!/usr/bin/python

import sys
import pickle
sys.path.append("../Other/tools/ud120-projects-master/tools")

from feature_format import featureFormat, targetFeatureSplit
from tester import dump_classifier_and_data

### Task 1: Select what features you'll use.
### features_list is a list of strings, each of which is a feature name.
# You will need to use more features
### The first feature must be "poi".
features_list = ['poi','salary' , 'deferral_payments', 'total_payments', 'loan_advances',
                 'bonus', 'restricted_stock_deferred', 'deferred_income', 'total_stock_value',
                 'expenses', 'exercised_stock_options', 'other', 'long_term_incentive',
                 'restricted_stock', 'director_fees', 'to_messages', 'from_poi_to_this_person',
                 'from_messages', 'from_this_person_to_poi', 'shared_receipt_with_poi']

### Load the dictionary containing the dataset
with open("final_project_dataset.pkl", "rb") as data_file:
    data_dict = pickle.load(data_file)

import matplotlib.pyplot as plt
import numpy as np
#Check all data types
#for typ in data_dict['TOTAL']:
#    print(type(data_dict['TOTAL'][typ]))

#Plotting histograms
i=0
#for ftr in features_list:
#    print(ftr)
sal_arr=featureFormat(data_dict, ['salary'])
bon_arr=featureFormat(data_dict, ['bonus'])
tp_arr=featureFormat(data_dict, ['total_payments'])
tsv_arr=featureFormat(data_dict, ['total_stock_value'])
#plt.hist(sal_arr, 70)
#plt.hist(bon_arr, 100)
#plt.hist(tp_arr, 70)
plt.hist(tsv_arr, 70)
#
#Check for any NaNs in required fields such as email_address
for rec in data_dict:
    if data_dict[rec]['email_address'] == 'NaN':
        if data_dict[rec]['salary']=='NaN':
            print (rec)
#print(data_dict['THE TRAVEL AGENCY IN THE PARK'])

### Task 2: Remove outliers
def Plots(data_dict, x, y):
    """ Plot with flag = True in Red """
    data = featureFormat(data_dict, [x, y, 'poi'])
    for point in data:
        x = point[0]
        y = point[1]
        poi = point[2]
        if poi:
            color = 'red'
        else:
            color = 'blue'
        plt.scatter(x, y, color=color)
    plt.xlabel(x)
    plt.ylabel(y)
    plt.show()
#print(Plots(data_dict, 'total_payments', 'total_stock_value'))
#print(Plots(data_dict, 'from_poi_to_this_person', 'from_this_person_to_poi'))
print(Plots(data_dict, 'salary', 'bonus'))

data_dict.pop( 'THE TRAVEL AGENCY IN THE PARK', 0 )
data_dict.pop( 'TOTAL', 0 )
nan_cnt_ss=0
nan_cnt_emp_poi=0


### Task 3: Create new feature(s)
### Store to my_dataset for easy export below.
my_dataset = data_dict
for person in my_dataset:
    tsv = my_dataset[person]['total_stock_value']
    sal = my_dataset[person]['salary']
    if tsv != "NaN" and sal != "NaN":
        my_dataset[person]['stock_salary']= tsv/float(sal)
    else:
        my_dataset[person]['stock_salary']= 0
                  
    msg_from_poi = my_dataset[person]['from_poi_to_this_person']
    to_msg = my_dataset[person]['to_messages']
    if msg_from_poi != "NaN" and to_msg != "NaN":
        my_dataset[person]['msg_from_poi_ratio'] = msg_from_poi/float(to_msg)
    else:
        my_dataset[person]['msg_from_poi_ratio'] = 0
    msg_to_poi = my_dataset[person]['from_this_person_to_poi']
    from_msg = my_dataset[person]['from_messages']
    if msg_to_poi != "NaN" and from_msg != "NaN":
        my_dataset[person]['msg_to_poi_ratio'] = msg_to_poi/float(from_msg)
    else:
        my_dataset[person]['msg_to_poi_ratio'] = 0

    if msg_to_poi != "NaN" and msg_from_poi != "NaN" and from_msg != "NaN" and to_msg!="NaN":
        poi_interact = msg_from_poi + msg_to_poi
        total_interact = to_msg + from_msg
    if (poi_interact != "NaN" or poi_interact != "NaNNaN")and total_interact != "NaN":
        my_dataset[person]['emp_poi_interact'] = poi_interact/float(total_interact)
    else:
        my_dataset[person]['emp_poi_interact'] = 0

new_features_list = features_list + ['stock_salary', 'msg_to_poi_ratio', \
'msg_from_poi_ratio', 'emp_poi_interact']

## Extract features and labels from dataset for local testing
data = featureFormat(my_dataset, new_features_list, sort_keys = True)
labels, features = targetFeatureSplit(data)

#Select the best features:
#Removes all features whose variance is below 80%
from sklearn.feature_selection import VarianceThreshold
sel = VarianceThreshold(threshold=(.8 * (1 - .8)))
features = sel.fit_transform(features)

#Removes all but the k highest scoring features
from sklearn.feature_selection import f_classif
from sklearn.feature_selection import SelectKBest
from sklearn import preprocessing

k = 7
selector = SelectKBest(f_classif, k=7)
selector.fit_transform(features, labels)
print("Best features:")
scores = zip(new_features_list[1:],selector.scores_)
sorted_scores = sorted(scores, key = lambda x: x[1], reverse=True)
print (sorted_scores)
optimized_features_list = ['poi'] + list(map(lambda x: x[0], sorted_scores))[0:k]
print("Optimized features:")
print(optimized_features_list)

# Extract from dataset without new features
data = featureFormat(my_dataset, optimized_features_list, sort_keys = True)
labels, features = targetFeatureSplit(data)
scaler = preprocessing.MinMaxScaler()
features = scaler.fit_transform(features)
# Extract from dataset with new features
data = featureFormat(my_dataset, optimized_features_list + \
                     ['msg_to_poi_ratio', 'msg_from_poi_ratio'] + \
                     ['emp_poi_interact', 'stock_salary'], 
                     sort_keys = True)
new_f_labels, new_f_features = targetFeatureSplit(data)
new_f_features = scaler.fit_transform(new_f_features)

### Task 4: Try a varity of classifiers
### Please name your classifier clf for easy export below.
### Note that if you want to do PCA or other multi-stage operations,
### you'll need to use Pipelines. For more info:
### http://scikit-learn.org/stable/modules/pipeline.html

# Provided to give you a starting point. Try a variety of classifiers.
from sklearn.naive_bayes import GaussianNB
nb_clf = GaussianNB()

from sklearn.linear_model import LogisticRegression
l_clf = LogisticRegression(C=10**18, tol=10**-21, solver='newton-cg', multi_class='multinomial', max_iter=100)

### K-means Clustering
from sklearn.cluster import KMeans
#k_clf = KMeans(copy_x=True, init='k-means++', max_iter=300, n_clusters=2, n_init=10,
#       n_jobs=1, precompute_distances='auto', random_state=None, tol=0.001,
#      verbose=0)
k_clf = KMeans(n_clusters=2, tol=0.001)
#k_clf = KMeans(init='random', max_iter=300, n_clusters=2, n_init=10, tol=0.001)

### Adaboost Classifier
from sklearn.ensemble import AdaBoostClassifier
a_clf = AdaBoostClassifier(algorithm='SAMME')

### Support Vector Machine Classifier
from sklearn.svm import SVC
s_clf = SVC(kernel='rbf', C=100)

### Random Forest
from sklearn.ensemble import RandomForestClassifier
rf_clf = RandomForestClassifier(max_depth = 5,max_features = 'sqrt',n_estimators = 10, random_state = 42)

### Stochastic Gradient Descent - Logistic Regression
from sklearn.linear_model import SGDClassifier
g_clf = SGDClassifier(loss='log')

from numpy import mean
from sklearn import cross_validation
from sklearn.metrics import accuracy_score, precision_score, recall_score
def evaluate_clf(clf, features, labels, num_iters=100, test_size=0.3):
    print (clf)
    accuracy = []
    precision = []
    recall = []
    first = True
    for trial in range(num_iters):
        features_train, features_test, labels_train, labels_test =\
            cross_validation.train_test_split(features, labels, test_size=test_size)
        clf.fit(features_train, labels_train)
        predictions = clf.predict(features_test)
        accuracy.append(accuracy_score(labels_test, predictions))
        precision.append(precision_score(labels_test, predictions))
        recall.append(recall_score(labels_test, predictions))
        if trial % 10 == 0:
            if first:
                sys.stdout.write('\nProcessing')
            sys.stdout.write('.')
            sys.stdout.flush()
            first = False

    print ("done.\n")
    print ("precision: {}".format(mean(precision)))
    print ("recall:    {}".format(mean(recall)))
    return mean(precision), mean(recall)

#evaluate_clf(nb_clf, features, labels)
#evaluate_clf(l_clf, features, labels)
evaluate_clf(k_clf, features, labels)
#evaluate_clf(a_clf, features, labels) #AdaBoost
#evaluate_clf(s_clf, features, labels)
#evaluate_clf(rf_clf, features, labels) #RForest
#evaluate_clf(g_clf, features, labels)

### Task 5: Tune your classifier to achieve better than .3 precision and recall
### using our testing script. Check the tester.py script in the final project
### folder for details on the evaluation method, especially the test_classifier
### function. Because of the small size of the dataset, the script uses
### stratified shuffle split cross validation. For more info:
### http://scikit-learn.org/stable/modules/generated/sklearn.cross_validation.StratifiedShuffleSplit.html

# Create training sets and test sets
from sklearn.cross_validation import train_test_split, StratifiedShuffleSplit
from sklearn.grid_search import GridSearchCV

features_train, features_test, labels_train, labels_test = \
    train_test_split(features, labels, test_size=0.3, random_state=42)

# Cross-validation for parameter tuning in grid search
sss = StratifiedShuffleSplit(
    labels_train,
    n_iter = 20,
    test_size = 0.5,
    random_state = 0
    )
parameters = dict(
        #n_clusters = [2, 4],
        max_iter = [300, 500],
        n_init = [10, 50],
        init =['k-means++', 'random'],
        tol=[0.001, 0.0001]
        )

grid = GridSearchCV(k_clf, param_grid=parameters, cv=sss)
grid.fit(features_train, labels_train)
labels_pred = grid.predict(features_test)
print("The best parameters are %s with a score of %0.2f"
      % (grid.best_params_, grid.best_score_))

print ("New Acc Score: ", accuracy_score(labels_test, labels_pred))
print ("New Prec Score: ", precision_score(labels_test, labels_pred))
print ("New Rec Score: ", recall_score(labels_test, labels_pred))

### Task 6: Dump your classifier, dataset, and features_list so anyone can
### check your results. You do not need to change anything below, but make sure
### that the version of poi_id.py that you submit can be run on its own and
### generates the necessary .pkl files for validating your results.

dump_classifier_and_data(grid, my_dataset, optimized_features_list)