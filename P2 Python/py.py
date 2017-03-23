# -*- coding: utf-8 -*-
"""
Created on Sat Jan  7 21:34:46 2017

@author: mnshr
"""

#%%
import unicodecsv as uc
import numpy as np
import pandas as pd

file_nm = 'titanic-data.csv'
titanic_data = pd.read_csv(file_nm)


def convert_gender(str):
    if str=='male':
        return 0
    else:
        return 1

#titanic_data['Gender']=titanic_data['Sex'].apply(convert_gender)

def convert_emb (str):
    if str=='C':
        return 0
    elif str=='Q':
        return 1
    else:
        return 2
#titanic_data['Embarked_int']=titanic_data['Embarked'].apply(convert_emb)
#titanic_data = titanic_data.drop(['PassengerId','Name','Ticket','Sex', 'Embarked', 'Cabin'], axis=1)

print(titanic_data.groupby('Sex'))
titanic_data['Gender']=titanic_data['Sex'].apply(convert_gender)
titanic_data['Embarked_int']=titanic_data['Embarked'].apply(convert_emb)
#titanic_data.head(10)
#%%
median_ages = np.zeros((2,3))
for i in range(0, 2):
    for j in range(0, 3):
        median_ages[i,j] = titanic_data[(titanic_data['Gender'] == i) & \
                              (titanic_data['Pclass'] == j+1)]['Age'].dropna().median()
median_ages
titanic_data['AgeNaN']=titanic_data['Age']
#Fill the Age column
for i in range(0, 2):
    for j in range(0, 3):
        titanic_data.loc[ (titanic_data.Age.isnull()) & (titanic_data.Gender == i) & (titanic_data.Pclass == j+1),\
                'Age'] = median_ages[i,j]
titanic_data = titanic_data.drop(['PassengerId','Name','Ticket','Sex', 'Embarked', 'Cabin'], axis=1)
titanic_data.describe()
#%%
import seaborn as sns
import matplotlib.pyplot as plt
sns.set_style('whitegrid')

#sns.factorplot('Embarked','Survived', data=titanic_data,size=4,aspect=3)

titanic_data.describe()

#Gender Analysis
fig, (axis1,axis2) = plt.subplots(1,2,figsize=(10,5))
sns.countplot(x='Gender', data=titanic_data, ax=axis1)
person_perc = titanic_data[["Gender", "Survived"]].groupby(['Gender'],as_index=False).mean()
person_perc
sns.barplot(x='Gender', y='Survived', data=person_perc, ax=axis2)
#The plots show that there are more males, but on average more females survived
#%%
# Age analysis
# get average, std in titanic_data
average_age_titanic   = titanic_data["Age"].mean()
std_age_titanic       = titanic_data["Age"].std()

# plot Age values
fig, axis1 = plt.subplots(1, figsize=(15,4))
axis1.set_title('Age values - Titanic')
titanic_data['Age'].astype(int).hist(bins=70, ax=axis1)

# peaks for survived/not survived passengers by their age
facet = sns.FacetGrid(titanic_data, hue="Survived",aspect=4)
facet.map(sns.kdeplot,'Age',shade= True)
facet.set(xlim=(0, titanic_data['Age'].max()))
facet.add_legend()

# average survived passengers by age
fig, axis1 = plt.subplots(1,1,figsize=(18,4))
average_age = titanic_data[["Age", "Survived"]].groupby(['Age'],as_index=False).mean()
sns.barplot(x='Age', y='Survived', data=average_age)
#The plots above show that a higher number of survivors are children
#%%
#Family
#The following graphs will analyze survival from the perspective of the presenece
# of family members. Hence we combine the Parch and Sibsp to create a Family
# variable
titanic_data['Family']=titanic_data['Parch'] + titanic_data['SibSp']
#titanic_grp = titanic_data[['Family', 'Survived']].groupby(['Family'])

#fig, (axis1,axis2) = plt.subplots(1,2,sharex=True,figsize=(10,5))


titanic_data['Family'].describe()
#And here is the histogram for family
titanic_data['Family'].hist(bins=10)

#The graph below shows that the survival rate for families is higher than singletons
sns.countplot(x='Family',hue='Survived', data=titanic_data)#, ax=axis1)

#%%
from pandas import Series,DataFrame

#Comparing PClass and Fares

# plot figsize=(15,3),
#titanic_data['Fare'].plot(kind='hist', bins=100, xlim=(0,50))

# get fare for survived & didn't survive passengers
fare_not_survived = titanic_data["Fare"][titanic_data["Survived"] == 0]
fare_survived     = titanic_data["Fare"][titanic_data["Survived"] == 1]

# get average and std for fare of survived/not survived passengers
avg_fare = DataFrame([fare_not_survived.mean(), fare_survived.mean()])
std_fare      = DataFrame([fare_not_survived.std(), fare_survived.std()])

avg_fare.index.names = std_fare.index.names = ["Survived"]
avg_fare.plot(yerr=std_fare,kind='bar',legend=False)

ax =plt.axes()
ax.set_title('Gender and Fares')
#sns.boxplot(x="Pclass", y="Fare", hue='Survived', data=titanic_data, ax=axis1)

pc_gender = sns.countplot(x="Pclass", hue="Gender", data=titanic_data.sort_values(by="Pclass"))
pc_gender.set(xlabel='Passenger Class', ylabel='Number of passengers')

#%%
#Checking the correlations:
def survive_corr(col):
    corr = titanic_data[col].corr(titanic_data['Survived'])
    return pd.Series(corr, index=[col])

age_corr = survive_corr ('Age')
gender_corr = survive_corr ('Gender')
par_corr = survive_corr ('Parch')
sib_corr = survive_corr ('SibSp')
class_corr = survive_corr ('Pclass')
fare_corr = survive_corr('Fare')

corr_vector = [age_corr, gender_corr, par_corr, sib_corr, class_corr, fare_corr]
corr_vector
#%%
ag = titanic_data[["Age", "Survived"]].groupby('Age')
print(ag)