# -*- coding: utf-8 -*-
"""
Created on Wed Nov 30 13:31:52 2016

@author: yny424
"""

import random
import unicodecsv as uc
from datetime import datetime as dt
import numpy as np

"""
for x in range(20):
    (random.randint(1,2000))
print

print random.randrange(5,60, 5)
"""
#Date parsing
print (dt.strptime('2016-12-03', '%Y-%m-%d'))
enrollments = []
enrollments1 = []
#Integer parsing from string
print (int('32') * int(float('2')))


with open('enrollments.csv', 'rb')   as f: #check filename
    #if with open() is used then no need to call f.close()
    reader = uc.DictReader(f)
    #csv reader reads as string and data types have to be converted
    #enrollments1=list(reader) #no need to iterate over a for loop
    for row in reader:  #We can iterate over a loop only once
            enrollments.append(row)
     
with open('enrollments.csv', 'rb')   as f: #check filename
    #if with open() is used then no need to call f.close()
    reader = uc.DictReader(f)
    #csv reader reads as string and data types have to be converted
    enrollments1=list(reader) #no need to iterate over a for loop
    f.close()

print(enrollments1[0])
print(enrollments[1])

print (len(enrollments))
unique_enrolled_students = set() #Set has only unique elements
for enrollment in enrollments:
    unique_enrolled_students.add(enrollment['account_key'])
len(unique_enrolled_students)

print (len(unique_enrolled_students))

#change the keyname in the list, 
#acct_key is added, account_key still exists 
for en in enrollments:
    en['acct_key']=en['account_key']
    del[en['acct_key']] #deleting what i added

print (enrollments[0])

def removekey(d, key):
        r=dict(d)
        del r[key]
        return r

#%%
import numpy as np
# First 20 countries with employment data
countries = np.array([
    'Afghanistan', 'Albania', 'Algeria', 'Angola', 'Argentina',
    'Armenia', 'Australia', 'Austria', 'Azerbaijan', 'Bahamas',
    'Bahrain', 'Bangladesh', 'Barbados', 'Belarus', 'Belgium',
    'Belize', 'Benin', 'Bhutan', 'Bolivia',
    'Bosnia and Herzegovina'
])

# Employment data in 2007 for those 20 countries
employment = np.array([
    55.70000076,  51.40000153,  50.5       ,  75.69999695,
    58.40000153,  40.09999847,  61.5       ,  57.09999847,
    60.90000153,  66.59999847,  60.40000153,  68.09999847,
    66.90000153,  53.40000153,  48.59999847,  56.79999924,
    71.59999847,  58.40000153,  70.40000153,  41.20000076
])

def max_employment(countries, employment):
    '''
    Fill in this function to return the name of the country
    with the highest employment in the given employment
    data, and the employment in that country.
    '''
    max_value=0
    print(len(employment))
    for i in range(len(employment)):
        if max_value<employment[i]:
            max_value=employment[i]
            max_country=countries[i]
        i=i+1
    max_country1=countries[employment.argmax()]
    max_value1=employment.max()
    print(max_country1, max_value1)
    return (max_country, max_value)
    
max_employment(countries, employment)
employment1 = np.array([
    1, 2, 3, 4,
    1, 2, 3, 4,
    1, 2, 3, 4,
    1, 2, 3, 4,
    1, 2, 3, 4
])
#Vectorized Addition
employment2=employment+employment1
print(employment2)

#List Concatenation
e1 = list([
    1, 2, 3, 4
])
e2 = list([
    5, 6, 7, 8
])
e=e1+e2
print(e)
#%%
#Index Arrays
import numpy as np
a = np.array([1, 2, 3, 2, 1])
b = (a >= 2)
print (b)
print (a[b])
print (a[a >= 2])

c = np.array([1, 2, 3, 4, 5])
d = np.array([1, 2, 3, 2, 1])

print (d == 2)
print (c[d == 2])

# Time spent in the classroom in the first week for 20 students
time_spent = np.array([
       12.89697233,    0.        ,   64.55043217,    0.        ,
       24.2315615 ,   39.991625  ,    0.        ,    0.        ,
      147.20683783,    0.        ,    0.        ,    0.        ,
       45.18261617,  157.60454283,  133.2434615 ,   52.85000767,
        0.        ,   54.9204785 ,   26.78142417,    0.
])

# Days to cancel for 20 students
days_to_cancel = np.array([
      4,   5,  37,   3,  12,   4,  35,  38,   5,  37,   3,   3,  68,
     38,  98,   2, 249,   2, 127,  35
])
def mean_time_for_paid_students(time_spent, days_to_cancel):
    print(days_to_cancel>=7)
    arr_m=time_spent[days_to_cancel>=7]
    print(arr_m.mean())
    return None
    
mean_time_for_paid_students(time_spent, days_to_cancel)

#%%
#+= (in place operation) updates the existing array while = + creates a new array
import numpy as np
a=np.array([1, 2, 3, 4])
b=a
a+=np.array([1,1,1,1])
print(b)
#%%