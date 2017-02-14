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
for typ in data_dict['TOTAL']:
    print(type(data_dict['TOTAL'][typ]))

#Plotting histograms
i=0
for ftr in features_list:
    print(ftr)
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
print(data_dict['THE TRAVEL AGENCY IN THE PARK'])

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
# Bonus-salary ratio | from person to poi | from poi to person
from collections import defaultdict
ft_sum=defaultdict(lambda:0)
ft_cnt=defaultdict(lambda:0)

for features in data_dict.items():
	if features[1]['total_stock_value'] == "NaN" or features[1]['salary'] == "NaN":
            nan_cnt_ss=nan_cnt_ss+1
            features[1]['stock_salary'] = "NaN"
	else:
         features[1]['stock_salary'] = float(features[1]['total_stock_value']) / float(features[1]['salary'])
for features in data_dict.items():
     print(features[1]['stock_salary'])
     print("------------")
        # stock_salary_sum=stock_salary_sum+features[1]['stock_salary']
        # stock_salary_cnt=stock_salary_cnt+1

features_list+=['stock_salary']

for emp, feat in data_dict.items():
    if feat['from_this_person_to_poi'] == "NaN" or feat['from_poi_to_this_person']  == "NaN" or feat['to_messages']  == "NaN" or feat['from_messages'] == "NaN" :
        nan_cnt_emp_poi=nan_cnt_emp_poi+1
        print("Setting NaN")
        feat['emp_poi_interact'] = "NaN"

    else:
        feat['emp_poi_interact']=(float(feat['from_this_person_to_poi']) + float(feat['from_poi_to_this_person']))/(float(feat['to_messages']) + float(feat['from_messages']))
        print(feat['emp_poi_interact'])
#    emp_poi_interact_cnt=emp_poi_interact_cnt+1
#    emp_poi_interact_sum=emp_poi_interact_sum+feat['emp_poi_interact']

#for emp, feat in data_dict.items():
#    if feat['emp_poi_interact'] == "NaN":
#        feat['emp_poi_interact'] = float(emp_poi_interact_sum/emp_poi_interact_cnt)

features_list+=['emp_poi_interact']

print("NaN SS: ", nan_cnt_ss, "NaN EPoi: ", nan_cnt_emp_poi)

#Remove NaNs
for emp, feat in data_dict.items():
    for ft in features_list:
        if feat[ft]!="NaN":
            ft_sum[ft]+=feat[ft]
            ft_cnt[ft]+=1
ft_mn={}
for ft in features_list:
    print(ft)
    ft_mn[ft]=float(ft_sum[ft]/ft_cnt[ft])

for emp, feat in data_dict.items():
    for ft in features_list:
        if feat[ft] == "NaN":
            feat[ft]= ft_mn[ft]

### Store to my_dataset for easy export below.
my_dataset = data_dict

##Fill the NaNs

### Extract features and labels from dataset for local testing
data = featureFormat(my_dataset, features_list, sort_keys = True)
labels, features = targetFeatureSplit(data)

from sklearn.preprocessing import MinMaxScaler
# Scale features
scaler = MinMaxScaler()
features = scaler.fit_transform(features)

from sklearn.feature_selection import SelectKBest
# K-best features
k_best = SelectKBest(k=5)
k_best.fit(features, labels)

results_list = zip(k_best.get_support(), features_list[1:], k_best.scores_)
results_list = sorted(results_list, key=lambda x: x[2], reverse=True)
print ("K-best features:", results_list)

data = featureFormat(my_dataset, features_list, sort_keys = True)
labels, features = targetFeatureSplit(data)
#%%
### Task 4: Try a varity of classifiers
### Please name your classifier clf for easy export below.
### Note that if you want to do PCA or other multi-stage operations,
### you'll need to use Pipelines. For more info:
### http://scikit-learn.org/stable/modules/pipeline.html

# Provided to give you a starting point. Try a variety of classifiers.
from sklearn.naive_bayes import GaussianNB
#clf = GaussianNB()

### Task 5: Tune your classifier to achieve better than .3 precision and recall
### using our testing script. Check the tester.py script in the final project
### folder for details on the evaluation method, especially the test_classifier
### function. Because of the small size of the dataset, the script uses
### stratified shuffle split cross validation. For more info:
### http://scikit-learn.org/stable/modules/generated/sklearn.cross_validation.StratifiedShuffleSplit.html

# Example starting point. Try investigating other evaluation techniques!
from sklearn.cross_validation import train_test_split
#features_train, features_test, labels_train, labels_test = \
#    train_test_split(features, labels, test_size=0.3, random_state=42)

### Task 6: Dump your classifier, dataset, and features_list so anyone can
### check your results. You do not need to change anything below, but make sure
### that the version of poi_id.py that you submit can be run on its own and
### generates the necessary .pkl files for validating your results.

#dump_classifier_and_data(clf, my_dataset, features_list)