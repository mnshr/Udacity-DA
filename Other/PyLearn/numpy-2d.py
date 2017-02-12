# -*- coding: utf-8 -*-
"""
Created on Mon Dec 12 05:29:14 2016

@author: Manish
"""

import numpy as np

# Subway ridership for 5 stations on 10 different days
ridership = np.array([
    [   0,    0,    2,    5,    0],
    [1478, 3877, 3674, 2328, 2539],
    [1613, 4088, 3991, 6461, 2691],
    [1560, 3392, 3826, 4787, 2613],
    [1608, 4802, 3932, 4477, 2705],
    [1576, 3933, 3909, 4979, 2685],
    [  95,  229,  255,  496,  201],
    [   2,    0,    1,   27,    0],
    [1438, 3785, 3589, 4174, 2215],
    [1342, 4043, 4009, 4665, 3033]
])
print (ridership[1, 3])
print (ridership[1:3, 3:1])
print (ridership[1, :3])
#%%
import matplotlib.pyplot as plt
 # Generate some weird data with hugely unscaled features
rng = np.random.RandomState(0)
n_samples = 100
n_features = 10

X = rng.normal(size=(n_samples, n_features))
#print (X)
Y=X
Y[:, :2] *= 1e300
print (Y-X)
#%%
import unicodecsv as uc
import numpy as np
def mean_riders_for_max_station(ridership):

    '''
    Fill in this function to find the station with the maximum riders on the
    first day
    '''
    print(np.mean(ridership[:,np.argmax(ridership[0,:])]))
    print(np.mean(ridership))
    print(ridership.mean(axis=0)) #mean per station (column)
    print(ridership.mean(axis=1)) #mean per day (row)
    overall_mean = 0 #None # Replace this with your code
    mean_for_max = 0 #None # Replace this with your code
    print(min(ridership.mean(axis=1))) #min per day (row)
    print(max(ridership.mean(axis=1))) #max per day (row)
    return (overall_mean, mean_for_max)

mean_riders_for_max_station(ridership)

#%%
#Axis argument
a = np.array([
        [1, 2, 3],
        [4, 5, 6],
        [7, 8, 9]
    ])

print (a.sum())
print (a.sum(axis=0))
print (a.sum(axis=1))

ridership = np.array([[ 5, 10, 15, 20, 25],
       [ 5, 10, 15, 20, 25],
       [ 5, 10, 15, 20, 25]])
print(min(ridership.mean(axis=0))) #min per station (col)
print(max(ridership.mean(axis=0))) #max per station (col)

#%%
#Data Frame - Each col is assumed to be a different type

import pandas as pd

# Subway ridership for 5 stations on 10 different days
ridership_df = pd.DataFrame(
    data=[[   0,    0,    2,    5,    0],
          [1478, 3877, 3674, 2328, 2539],
          [1613, 4088, 3991, 6461, 2691],
          [1560, 3392, 3826, 4787, 2613],
          [1608, 4802, 3932, 4477, 2705],
          [1576, 3933, 3909, 4979, 2685],
          [  95,  229,  255,  496,  201],
          [   2,    0,    1,   27,    0],
          [1438, 3785, 3589, 4174, 2215],
          [1342, 4043, 4009, 4665, 3033]],
    index=['05-01-11', '05-02-11', '05-03-11', '05-04-11', '05-05-11',
           '05-06-11', '05-07-11', '05-08-11', '05-09-11', '05-10-11'],
    columns=['R003', 'R004', 'R005', 'R006', 'R007']
)
print(ridership_df)
print (ridership_df.iloc[0]) #prints 1st row
print ('--------')
print (ridership_df.loc['05-05-11']) #prints 5th ro
print ('--------')
print (ridership_df['R003']) #print 1st col
print ('--------')
print (ridership_df.iloc[1, 3])
print ('--------')
print (ridership_df.iloc[1:4]) #prints 3 rows from 2nd
print ('--------')
print (ridership_df[['R003', 'R005']]) #Prints 2 columns
print ('--------')
print(np.argmax(ridership_df.iloc[0]))
print(ridership_df.iloc[0].argmax()) #Returns the column name of DF (index)

def mean_riders_for_max_station(ridership):
    max_stn=np.argmax(ridership.iloc[0])
    overall_mean = ridership.values.mean() # Overall Mean
    mean_for_max = ridership[max_stn].mean() # Mean for station with max riders on 1st day
    print(overall_mean)
    print(mean_for_max)
    return (overall_mean, mean_for_max)

mean_riders_for_max_station(ridership_df)

#%%
import pandas as pd
#Correlation
filename = 'nyc_subway_weathe.csv'
subway_df = pd.read_csv(filename)
entries = subway_df['ENTRIESn_hourly']
cum_entries = subway_df['ENTRIESn']
rain = subway_df['meanprecipi']
temp = subway_df['meantempi']

def correlation(x, y): #Pearson's R
    """
    correlation = average of (x in standard units) times (y in standard units)
    Remember to pass the argument "ddof=0" to the Pandas std() function!
    """
    std_x=(x-x.mean())/x.std(ddof=0) #ddof=0 to not apply Bessel's correction
    std_y=(y-y.mean())/y.std(ddof=0)
    return (std_x*std_y).mean()

print (correlation(entries, rain))
print (correlation(entries, temp))
print (correlation(rain, temp))

print (correlation(entries, cum_entries))

#%%
#Shift function in Pandas DF
# Cumulative entries and exits for one station for a few hours.
entries_and_exits = pd.DataFrame({
    'ENTRIESn': [3144312, 3144335, 3144353, 3144424, 3144594,
                 3144808, 3144895, 3144905, 3144941, 3145094],
    'EXITSn': [1088151, 1088159, 1088177, 1088231, 1088275,
               1088317, 1088328, 1088331, 1088420, 1088753]
})

def get_hourly_entries_and_exits(entries_and_exits):
    '''
    Fill in this function to take a DataFrame with cumulative entries
    and exits (entries in the first column, exits in the second) and
    return a DataFrame with hourly entries and exits (entries in the
    first column, exits in the second).
    '''
    return entries_and_exits - entries_and_exits.shift(1)

print(get_hourly_entries_and_exits(entries_and_exits))
print(entries_and_exits.shift(1))
#%%
import pandas as pd
#Apply and Applymap() functions
grades_df = pd.DataFrame(
    data={'exam1': [43, 81, 78, 75, 89, 70, 91, 65, 98, 87],
          'exam2': [24, 63, 56, 56, 67, 51, 79, 46, 72, 60]},
    index=['Andre', 'Barry', 'Chris', 'Dan', 'Emilio',
           'Fred', 'Greta', 'Humbert', 'Ivan', 'James']
)

def convert_grades(grades):
    if grades >=90:
        return 'A'
    elif grades >=70 & grades <80:
        return 'C'
    elif grades >=80 & grades <90:
        return 'B'
    elif grades >=60 & grades <70:
        return 'D'
    else :
        return 'F'

print(convert_grades(39))
#Apply works on a single piece (may be an index, a column or an element)

grades_df = pd.DataFrame(
    data={'exam1': [43, 81, 78, 75, 89, 70, 91, 65, 98, 87],
          'exam2': [24, 63, 56, 56, 67, 51, 79, 46, 72, 60]},
    index=['Andre', 'Barry', 'Chris', 'Dan', 'Emilio',
           'Fred', 'Greta', 'Humbert', 'Ivan', 'James']
)

def convert_grades_curve(exam_grades):
        # Pandas has a bult-in function that will perform this calculation
        # This will give the bottom 0% to 10% of students the grade 'F',
        # 10% to 20% the grade 'D', and so on. You can read more about
        # the qcut() function here:
        # http://pandas.pydata.org/pandas-docs/stable/generated/pandas.qcut.html
        return pd.qcut(exam_grades,
                       [0, 0.1, 0.2, 0.5, 0.8, 1],
                       labels=['F', 'D', 'C', 'B', 'A'])

# qcut() operates on a list, array, or Series. This is the
# result of running the function on a single column of the
# DataFrame.
print (convert_grades_curve(grades_df['exam1']))

# qcut() does not work on DataFrames, but we can use apply()
# to call the function on each column separately
print (grades_df.apply(convert_grades_curve))
def standardize_column(column):
    return (column-column.mean())/column.std()
def standardize(df):
    return df.apply(standardize_column)

print(standardize(grades_df)    )

#%%
import numpy as np
import pandas as pd
#Apply to work on a column and return a single value
df = pd.DataFrame({
    'a': [4, 5, 3, 1, 2],
    'b': [20, 10, 40, 50, 30],
    'c': [25, 20, 5, 15, 10]
})

print (df.apply(np.mean))
print (df.apply(np.max))

def sec_largest(col):
    sorted_col=col.sort_values(ascending=False)
    return sorted_col.iloc[1]

print('////////////')
def second_largest(df):
    return df.apply(sec_largest)

print(second_largest(df))
#%%
import pandas as pd

# Adding a Series to a square DataFrame
print ('-------Adding a Series to a Square DataFrame---------------')
s = pd.Series([1, 2, 3, 4])
df = pd.DataFrame({
    0: [10, 20, 30, 40],
    1: [50, 60, 70, 80],
    2: [90, 100, 110, 120],
    3: [130, 140, 150, 160]
})

print (df)
print ('') # Create a blank line between outputs
print (df + s)
print ('-------Adding a Series to a one-row DataFrame---------------')
# Adding a Series to a one-row DataFrame
s = pd.Series([1, 2, 3, 4])
df = pd.DataFrame({0: [10], 1: [20], 2: [30], 3: [40]})

print (df)
print ('') # Create a blank line between outputs
print (df + s)
print ('------Adding a Series to a one-column DataFrame----------------')
# Adding a Series to a one-column DataFrame
s = pd.Series([1, 2, 3, 4])
df = pd.DataFrame({0: [10, 20, 30, 40]})

print (df)
print ('') # Create a blank line between outputs
print (df + s) #WITH NANS
print(df.add(s, axis='columns')) #WITH NANS
print(df.add(s, axis='index')) #NO NANS
print ('----------Adding with axis=index------------')

# Adding with axis='index'
s = pd.Series([1, 2, 3, 4])
df = pd.DataFrame({
    0: [10, 20, 30, 40],
    1: [50, 60, 70, 80],
    2: [90, 100, 110, 120],
    3: [130, 140, 150, 160]
})

print (df)
print (' ') # Create a blank line between outputs
print (df.add(s, axis='index'))
# The functions sub(), mul(), and div() work similarly to add()
print ('------Adding when DataFrame column names match Series index----------------')

# Adding when DataFrame column names match Series index
s = pd.Series([1, 2, 3, 4], index=['a', 'b', 'e', 'd'])
df = pd.DataFrame({
    'a': [10, 20, 30, 40],
    'b': [50, 60, 70, 80],
    'c': [90, 100, 110, 120],
    'd': [130, 140, 150, 160]
})
#C and E are not matching columns, hence NaNs for them

print (df)
print ('') # Create a blank line between outputs
print (df + s)
print ('------DataFrame column names dont match Series index----------------')
# Adding when DataFrame column names don't match Series index
s = pd.Series([1, 2, 3, 4])
df = pd.DataFrame({
    'a': [10, 20, 30, 40],
    'b': [50, 60, 70, 80],
    'c': [90, 100, 110, 120],
    'd': [130, 140, 150, 160]
})

print (df)
print ('') # Create a blank line between outputs
print (df + s)
print ('----------------------')

#%%
grades_df = pd.DataFrame(
    data={'exam1': [43, 81, 78, 75, 89, 70, 91, 65, 98, 87],
          'exam2': [24, 63, 56, 56, 67, 51, 79, 46, 72, 60]},
    index=['Andre', 'Barry', 'Chris', 'Dan', 'Emilio',
           'Fred', 'Greta', 'Humbert', 'Ivan', 'James']
)
def standardize(df):
    return (df-df.mean())/df.std()

print(standardize(grades_df))
def standardize_rows(df):
     return(df.sub(df.mean(axis='columns'), axis='index')/
     df.div(df.std(axis='columns'), axis='index'))

print(standardize_rows(grades_df))
#%% Group by
#df.groupby('key').sum()['param'].mean()
#df.groupby('key').groups - To examine the underlying groups

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

values = np.array([1, 3, 2, 4, 1, 6, 4])
example_df = pd.DataFrame({
    'value': values,
    'even': values % 2 == 0,
    'above_three': values > 3
}, index=['a', 'b', 'c', 'd', 'e', 'f', 'g'])


print(example_df)
grouped_data = example_df.groupby('even')
# The groups attribute is a dictionary mapping keys to lists of row indexes
print (grouped_data.groups)

grouped_data = example_df.groupby(['even', 'above_three'])
print (grouped_data.groups)

grouped_data = example_df.groupby('even')
print (grouped_data.sum())

grouped_data = example_df.groupby('even')

# You can take one or more columns from the result DataFrame
print (grouped_data.sum()['value'])

print ('\n') # Blank line to separate results

# You can also take a subset of columns from the grouped data before
# collapsing to a DataFrame. In this case, the result is the same.
print (grouped_data['value'].sum())
#%%
#Groupby using nyc_subway data
import pandas as pd

df = pd.read_csv("nyc_subway_weathe.csv")
print(df.head())

ridershipby_day=df.groupby('day_week').mean()['ENTRIESn_hourly']


import seaborn as sns
ridershipby_day.plot()
#%%
import numpy as np
import pandas as pd

values = np.array([1, 3, 2, 4, 1, 6, 4])
example_df = pd.DataFrame({
    'value': values,
    'even': values % 2 == 0,
    'above_three': values > 3
}, index=['a', 'b', 'c', 'd', 'e', 'f', 'g'])

def standardize(xs):
    return (xs - xs.mean()) / xs.std()

print(example_df)
grouped_data = example_df.groupby('even')
print(grouped_data)
print (grouped_data['value'].apply(standardize))
print(example_df.describe())
print('----------------------------------------')
# DataFrame with cumulative entries and exits for multiple stations
ridership_df = pd.DataFrame({
    'UNIT': ['R051', 'R079', 'R051', 'R079', 'R051', 'R079', 'R051', 'R079', 'R051'],
    'TIMEn': ['00:00:00', '02:00:00', '04:00:00', '06:00:00', '08:00:00', '10:00:00', '12:00:00', '14:00:00', '16:00:00'],
    'ENTRIESn': [3144312, 8936644, 3144335, 8936658, 3144353, 8936687, 3144424, 8936819, 3144594],
    'EXITSn': [1088151, 13755385,  1088159, 13755393,  1088177, 13755598, 1088231, 13756191,  1088275]
})

def gheae (e_e):
    return e_e - e_e.shift(1)

def get_hourly_entries_and_exits(entries_and_exits):
    station_grp=entries_and_exits.groupby('UNIT')
    print(station_grp)
    return (station_grp['ENTRIESn','EXITSn'].apply(gheae))

print(get_hourly_entries_and_exits(ridership_df))

#%%
#Merge is similar to SQL JOIN, with Inner/Left/Outer and Right arguments
import pandas as pd

subway_df = pd.DataFrame({
    'UNIT': ['R003', 'R003', 'R003', 'R003', 'R003', 'R004', 'R004', 'R004',
             'R004', 'R004'],
    'DATEn': ['05-01-11', '05-02-11', '05-03-11', '05-04-11', '05-05-11',
              '05-01-11', '05-02-11', '05-03-11', '05-04-11', '05-05-11'],
    'hour': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    'ENTRIESn': [ 4388333,  4388348,  4389885,  4391507,  4393043, 14656120,
                 14656174, 14660126, 14664247, 14668301],
    'EXITSn': [ 2911002,  2911036,  2912127,  2913223,  2914284, 14451774,
               14451851, 14454734, 14457780, 14460818],
    'latitude': [ 40.689945,  40.689945,  40.689945,  40.689945,  40.689945,
                  40.69132 ,  40.69132 ,  40.69132 ,  40.69132 ,  40.69132 ],
    'longitude': [-73.872564, -73.872564, -73.872564, -73.872564, -73.872564,
                  -73.867135, -73.867135, -73.867135, -73.867135, -73.867135]
})

weather_df = pd.DataFrame({
    'DATEn': ['05-01-11', '05-01-11', '05-02-11', '05-02-11', '05-03-11',
              '05-03-11', '05-04-11', '05-04-11', '05-05-11', '05-05-11'],
    'hour': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    'latitude': [ 40.689945,  40.69132 ,  40.689945,  40.69132 ,  40.689945,
                  40.69132 ,  40.689945,  40.69132 ,  40.689945,  40.69132 ],
    'longitude': [-73.872564, -73.867135, -73.872564, -73.867135, -73.872564,
                  -73.867135, -73.872564, -73.867135, -73.872564, -73.867135],
    'pressurei': [ 30.24,  30.24,  30.32,  30.32,  30.14,  30.14,  29.98,  29.98,
                   30.01,  30.01],
    'fog': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    'rain': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    'tempi': [ 52. ,  52. ,  48.9,  48.9,  54. ,  54. ,  57.2,  57.2,  48.9,  48.9],
    'wspdi': [  8.1,   8.1,   6.9,   6.9,   3.5,   3.5,  15. ,  15. ,  15. ,  15. ]
})

print (subway_df.head(3))
print (weather_df.head(3))
print (subway_df.merge(weather_df, 'inner'))

def combine_dfs(subway_df, weather_df):
    return subway_df.merge(weather_df,
                           left_on=['DATEn', 'hour', 'latitude', 'longitude'],
                           right_on=['date', 'hour', 'latitude', 'longitude'],
                           how='inner')

print ('====================')
#print (combine_dfs(subway_df, weather_df))

weather_df.plot()

#%%
import pandas as pd
import seaborn as sns

values = np.array([1, 3, 2, 4, 1, 6, 4])
example_df = pd.DataFrame({
    'value': values,
    'even': values % 2 == 0,
    'above_three': values > 3
}, index=['a', 'b', 'c', 'd', 'e', 'f', 'g'])

# Change False to True for this block of code to see what it does

# groupby() without as_index
first_even = example_df.groupby('even').first()
print (first_even)
#print (first_even['even']) # Causes an error.
#'even' is no longer a column in the DataFrame, they are row indexes now

print ('====================')
# groupby() with as_index=False

first_even = example_df.groupby('even', as_index=False).first()
print (first_even)
print (first_even['even']) # Now 'even' is still a column in the DataFrame

