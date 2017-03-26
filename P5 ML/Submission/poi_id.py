# -*- coding: utf-8 -*-
"""
Created on Mon Feb 20 07:49:51 2017

@author: mnshr
"""

#!/usr/bin/python

import sys
import pickle
import time
import matplotlib.pyplot as plt
from collections import defaultdict
from sklearn.feature_selection import VarianceThreshold
from sklearn.feature_selection import f_classif
from sklearn.feature_selection import SelectKBest
from sklearn import preprocessing

from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.cluster import KMeans
from sklearn.ensemble import AdaBoostClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import SGDClassifier
from numpy import mean
from sklearn import cross_validation
from sklearn.metrics import f1_score, make_scorer
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.cross_validation import train_test_split, StratifiedShuffleSplit
from sklearn.grid_search import GridSearchCV

sys.path.append("../Other/tools/ud120-projects-master/tools")

from feature_format import featureFormat, targetFeatureSplit
from tester import dump_classifier_and_data

### Task 1: Select what features you'll use.
### enron_features is a list of strings, each of which is a feature name.
# You will need to use more features
### The first feature must be "poi".
enron_features = ['poi','salary' , 'deferral_payments', 'total_payments', 'loan_advances',
                 'bonus', 'restricted_stock_deferred', 'deferred_income', 'total_stock_value',
                 'expenses', 'exercised_stock_options', 'other', 'long_term_incentive',
                 'restricted_stock', 'director_fees', 'to_messages', 'from_poi_to_this_person',
                 'from_messages', 'from_this_person_to_poi', 'shared_receipt_with_poi']

### Load the dictionary containing the dataset
with open("final_project_dataset.pkl", "rb") as data_file:
    enron_data = pickle.load(data_file)

#Check all data types
for typ in enron_data['TOTAL']:
    print(typ, ", ", type(enron_data['TOTAL'][typ]))

poi_cnt = 0
miss_sal_eml=0
#Dictionary to store the NaN counts of features
feature_nan_cnt = defaultdict(lambda:0)
#
#Check for any NaNs in apparently required fields such as email_address, salary
for rec, features in enron_data.iteritems():
    if enron_data[rec]['email_address'] == 'NaN':
        if enron_data[rec]['salary']=='NaN':
            #print (rec)
            miss_sal_eml +=1
    if enron_data[rec]['poi'] == True:
        poi_cnt+=1
    for fl in enron_features:
        #print enron_data[rec][fl]
        if fl == 'poi':
            continue
        #print features[fl]
        if enron_data[rec][fl] == 'NaN':
            #print enron_data[rec][fl]
            feature_nan_cnt[fl]+=1
        
print "Total number of PoI: ", poi_cnt
print "Number missing salary and email address: ", miss_sal_eml
print feature_nan_cnt; #All features have at least 20 NaNs

### Task 2: Remove outliers
def Plots(enron_data, x, y):
    """ Scatter Plotting numerical variables passed as arguments """
    data = featureFormat(enron_data, [x, y, 'poi'])
    for pt in data:
        x = pt[0]
        y = pt[1]
        poi = pt[2]
        if poi:
            color = 'orange'
        else:
            color = 'green'
        plt.scatter(x, y, color=color)
    plt.xlabel(x)
    plt.ylabel(y)
    plt.show()
print(Plots(enron_data, 'total_payments', 'total_stock_value'))
#print(Plots(enron_data, 'from_poi_to_this_person', 'from_this_person_to_poi'))
print(Plots(enron_data, 'salary', 'bonus'))

enron_data.pop( 'THE TRAVEL AGENCY IN THE PARK', 0 )
enron_data.pop( 'LOCKHART EUGENE E', 0)
enron_data.pop( 'TOTAL', 0 )

### Task 3: Create new feature(s)
### Copying the enron dictionary data to a temp dictionary for feature  engg
### Creating 4 new variables using the existing
tmp_enron_data = enron_data
for rec in tmp_enron_data:
    tsv = tmp_enron_data[rec]['total_stock_value']
    sal = tmp_enron_data[rec]['salary']
    if tsv != "NaN" and sal != "NaN":
        tmp_enron_data[rec]['stock_salary']= tsv/float(sal)
    else:
        tmp_enron_data[rec]['stock_salary']= 0

    msg_from_poi = tmp_enron_data[rec]['from_poi_to_this_person']
    to_msg = tmp_enron_data[rec]['to_messages']
    if msg_from_poi != "NaN" and to_msg != "NaN":
        tmp_enron_data[rec]['msg_from_poi'] = msg_from_poi/float(to_msg)
    else:
        tmp_enron_data[rec]['msg_from_poi'] = 0
    
    msg_to_poi = tmp_enron_data[rec]['from_this_person_to_poi']
    from_msg = tmp_enron_data[rec]['from_messages']
    if msg_to_poi != "NaN" and from_msg != "NaN":
        tmp_enron_data[rec]['msg_to_poi'] = msg_to_poi/float(from_msg)
    else:
        tmp_enron_data[rec]['msg_to_poi'] = 0

    if msg_to_poi != "NaN" and msg_from_poi != "NaN" and from_msg != "NaN" and to_msg!="NaN":
        poi_interact = msg_from_poi + msg_to_poi
        total_interact = to_msg + from_msg
    if (poi_interact != "NaN" or poi_interact != "NaN")and total_interact != "NaN":
        tmp_enron_data[rec]['emp_poi_interact'] = poi_interact/float(total_interact)
    else:
        tmp_enron_data[rec]['emp_poi_interact'] = 0

#Create a new dictionary with newly inttroduced featuds
enron_features_plus = enron_features + ['stock_salary', 'msg_to_poi', \
'msg_from_poi', 'emp_poi_interact']

## Impute NaNs and get the Numpy Array using featureFormat
data = featureFormat(tmp_enron_data, enron_features_plus, sort_keys = True)
labels, features = targetFeatureSplit(data)


#Select the best features:
#Removes features with variance below 80%
#http://scikit-learn.org/stable/modules/feature_selection.html#variance-threshold
sel = VarianceThreshold(threshold=(.8 * (1 - .8)))
features_var = sel.fit_transform(features)
features_var_lst = []
for item in features_var:
    features_var_lst.append(item)


# Extract from dataset without new features
#data = featureFormat(tmp_enron_data, enron_features_plus, sort_keys = True)
#labels, features = targetFeatureSplit(data)
scaler = preprocessing.MinMaxScaler()
features_np = scaler.fit_transform(features_var_lst)
features_scaled = []
for item in features_np:
    features_scaled.append(item)
    
#Removes all but the k highest scoring features
k = 7
selector = SelectKBest(f_classif, k=7)
selector.fit_transform(features_scaled, labels)
print("Best features:")
scores = zip(enron_features_plus[1:],selector.scores_)
sorted_scores = sorted(scores, key = lambda x: x[1], reverse=True)
print (sorted_scores)
optimized_enron_features = ['poi'] + list(map(lambda x: x[0], sorted_scores))[0:k]
print "Optimized features:", optimized_enron_features

# Extract from dataset with new features
data = featureFormat(tmp_enron_data, optimized_enron_features ,
                     sort_keys = True)
labels_opti, features_opti = targetFeatureSplit(data)
features_opti = scaler.fit_transform(features_opti)

### Task 4: Try a varity of classifiers
### Please name your classifier clf for easy export below.
### Note that if you want to do PCA or other multi-stage operations,
### you'll need to use Pipelines. For more info:
### http://scikit-learn.org/stable/modules/pipeline.html

def classifierPerf(clf, features, labels, num_iters=100, test_size=0.3):
    """
    This function evaluates a classifier and returns mean
    accuracy, precision, recall and F1 scores
    """
    print (clf)
    accuracy = []
    f_one = []
    precision = []
    recall = []
    print ('\nProcessing Starts at: ', time.time())
    for trial in range(num_iters):
        features_train, features_test, labels_train, labels_test =\
            cross_validation.train_test_split(features, labels, test_size=test_size)
        clf.fit(features_train, labels_train)
        pred = clf.predict(features_test)
        f_one.append(f1_score(labels_test, pred))
        accuracy.append(accuracy_score(labels_test, pred))
        precision.append(precision_score(labels_test, pred))
        recall.append(recall_score(labels_test, pred))

    print ("Processing ends at: ", time.time())
    print ("Accuracy:   {}".format(mean(accuracy)))
    print ("precision:  {}".format(mean(precision)))
    print ("recall:     {}".format(mean(recall)))
    print ("F1:         {}".format(mean(f_one)))
    return mean(accuracy), mean(precision), mean(recall), mean(f_one)

# Provided to give you a starting point. Try a variety of classifiers.
#Naive Bayes
nb_clf = GaussianNB()

#Logistic Regression
log_clf = LogisticRegression(C=10**18, tol=10**-21, solver='newton-cg', multi_class='multinomial', max_iter=100)

### K-means Clustering
#k_clf = KMeans(copy_x=True, init='k-means++', max_iter=300, n_clusters=2, n_init=10,
#       n_jobs=1, precompute_distances='auto', random_state=None, tol=0.001,
#      verbose=0)
km_clf = KMeans(n_clusters=2, tol=0.001)
#k_clf = KMeans(init='random', max_iter=300, n_clusters=2, n_init=10, tol=0.001)

### Adaboost Classifier
ab_clf = AdaBoostClassifier(algorithm='SAMME')

### Support Vector Machine Classifier
svc_clf = SVC(kernel='rbf', C=100)

### Random Forest
rf_clf = RandomForestClassifier(max_depth = 5,max_features = 'sqrt',n_estimators = 10, random_state = 42)

### Stochastic Gradient Descent - Logistic Regression
sgd_clf = SGDClassifier(loss='log')

#classifierPerf(nb_clf, features_opti, labels_opti) 
#classifierPerf(log_clf, features_opti, labels_opti)
classifierPerf(km_clf, features_opti, labels_opti)
#classifierPerf(ab_clf, features_opti, labels_opti)
#classifierPerf(svc_clf, features_opti, labels_opti)
#classifierPerf(rf_clf, features_opti, labels_opti) 
#classifierPerf(sgd_clf, features_opti, labels_opti)

#############################################################
### Task 5: Tune your classifier to achieve better than .3 precision and recall
### using our testing script. Check the tester.py script in the final project
### folder for details on the evaluation method, especially the test_classifier
### function. Because of the small size of the dataset, the script uses
### stratified shuffle split cross validation. For more info:
### http://scikit-learn.org/stable/modules/generated/sklearn.cross_validation.StratifiedShuffleSplit.html

features_train, features_test, labels_train, labels_test = \
    train_test_split(features_opti, labels_opti, test_size=0.3, random_state=42)

sss = StratifiedShuffleSplit(
    labels_train,
    n_iter = 20,
    test_size = 0.5,
    random_state = 0
    )
parameters = dict(
        n_clusters = [2, 4],
        max_iter = [300, 500],
        n_init = [10, 50],
        init =['k-means++', 'random'],
        tol=[0.001, 0.0001]
        )

grid = GridSearchCV(km_clf, param_grid=parameters, cv=sss)
grid.fit(features_train, labels_train)
labels_pred = grid.predict(features_test)
print("The best parameters are %s with a score of %0.2f"
      % (grid.best_params_, grid.best_score_))

#classifierPerf(k_clf, features_opti, labels_opti)
### Task 6: Dump your classifier, dataset, and enron_features so anyone can
### check your results. You do not need to change anything below, but make sure
### that the version of poi_id.py that you submit can be run on its own and
### generates the necessary .pkl files for validating your results.


dump_classifier_and_data(nb_clf, tmp_enron_data, optimized_enron_features)
#dump_classifier_and_data(log_clf, tmp_enron_data, optimized_enron_features)
#dump_classifier_and_data(km_clf, tmp_enron_data, optimized_enron_features)
#dump_classifier_and_data(ab_clf, tmp_enron_data, optimized_enron_features)
#dump_classifier_and_data(svc_clf, tmp_enron_data, optimized_enron_features)
#dump_classifier_and_data(rf_clf, tmp_enron_data, optimized_enron_features)
#dump_classifier_and_data(sgd_clf, tmp_enron_data, optimized_enron_features)
