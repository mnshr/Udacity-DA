# -*- coding: utf-8 -*-
"""
Created on Tue Jan 24 07:50:53 2017

@author: Manish
"""

from sklearn import linear_model
def studentReg(ages_train, net_worths_train):
    ### import the sklearn regression module, create, and train your regression
    ### name your regression reg

    ### your code goes here!
    reg = linear_model.Ridge (alpha = .5)
    # Or use this -> reg = LinearRegression()
    reg.fit (ages_train, net_worths_train)


    return reg

reg.predict(['Expects a list of values'])
reg.coef_ #Access the coefficients
reg.intercept_ #Access the intercept
reg.score(x_test, y_test) #Gives the r-squared score
#%%
from sklearn.cross_validation import train_test_split
#Split into training and testing data
#ages_train, ages_test, net_worths_train, net_worths_test =
#train_test_split(ages, net_worths)

import numpy
import random

def ageNetWorthData():

    random.seed(42)
    numpy.random.seed(42)

    ages = []
    for ii in range(100):
        ages.append( random.randint(20,65) )
    net_worths = [ii * 6.25 + numpy.random.normal(scale=40.) for ii in ages]
### need massage list into a 2d numpy array to get it to work in LinearRegression
    ages       = numpy.reshape( numpy.array(ages), (len(ages), 1))
    net_worths = numpy.reshape( numpy.array(net_worths), (len(net_worths), 1))

    from sklearn.cross_validation import train_test_split
    ages_train, ages_test, net_worths_train, net_worths_test = train_test_split(ages, net_worths)

    return ages_train, ages_test, net_worths_train, net_worths_test

import matplotlib.pyplot as plt


ages_train, ages_test, net_worths_train, net_worths_test = ageNetWorthData()

from sklearn.linear_model import LinearRegression

reg = LinearRegression()
reg.fit(ages_train, net_worths_train)

print (reg.predict([[27]][0][0]))
print(reg.coef_[0][0])
print(reg.intercept_[0])


### get the score on test data (THE R-SQ SCORE)
test_score = reg.score(ages_test, net_worths_test) ### fill in the line of code to get the right value


### get the r-squared score on the training data
training_score = reg.score(ages_train, net_worths_train) ### fill in the line of code to get the right value

print('Training score: ', training_score, ' | Testing Score: ', test_score)
plt.scatter(ages_train, net_worths_train)
plt.plot(ages_train, reg.predict(ages_train), color='blue', linewidth=3)
plt.xlabel('Ages')
plt.ylabel('Net Worth')