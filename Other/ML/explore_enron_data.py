#!/usr/bin/python

"""
    Starter code for exploring the Enron dataset (emails + finances);
    loads up the dataset (pickled dict of dicts).

    The dataset has the form:
    enron_data["LASTNAME FIRSTNAME MIDDLEINITIAL"] = { features_dict }

    {features_dict} is a dictionary of features associated with that person.
    You should explore features_dict as part of the mini-project,
    but here's an example to get you started:

    enron_data["SKILLING JEFFREY K"]["bonus"] = 5600000

"""

import pickle

enron_data = pickle.load(
open(
"../tools/ud120-projects-master/final_project/final_project_dataset.pkl", "rb"))

#print(enron_data)

j=0
k=0
l=0
m=0
n=0
o=0
p=0
for i in enron_data: #Finding the Persons of Interest
    print(i)
    if enron_data[i]['poi'] == 1:
        j=j+1;
    if enron_data[i]['salary']=='NaN':
        n=n+1
    if enron_data[i]['email_address']=='NaN':
        o=o+1
    if enron_data[i]['total_payments']=='NaN':
        p=p+1
    if i=='PRENTICE JAMES':
        k=enron_data[i]['total_stock_value']
    if i=='COLWELL WESLEY':
        l=enron_data[i]['from_messages']
    if i=='SKILLING JEFFREY K':
        m=enron_data[i]['exercised_stock_options']


print ('The PoI are: ', j)
print ('Total Stock val of Prentice James: ', k)
print ('From Messages of Wesley Colwell: ', l)
print ('ESOPs of Skilling: ', m)
print (enron_data['SKILLING JEFFREY K']['total_payments'])
print (enron_data['FASTOW ANDREW S']['total_payments'])
print (enron_data['LAY KENNETH L']['total_payments'])
print ('Quantifiable Sal: ', 146-n)
print ('Valid Email: ', 146-o)
print ('NaN in total_payments: ', p/146*100)