# -*- coding: utf-8 -*-
"""
Created on Thu Mar 16 19:24:02 2017

@author: Manish
"""

from datetime import datetime as dt
import pandas as pd


df = pd.read_csv("prosperLoanData.csv")


#print df['ListingKey']
#print df['LoanOriginationDate']
dt_df = pd.to_datetime(df['LoanOriginationDate'])

i=0
qt_df=[]

for v in dt_df:
    if dt_df[i].month <= 3:
        qt_df.append(str(dt_df[i].year)+'Q1')
        #print i, ' 1 ', qt_df
    elif dt_df[i].month > 3 and dt_df[i].month<=6:
        qt_df.append(str(dt_df[i].year)+'Q2')
        #print i,' 2', qt_df
    elif dt_df[i].month > 6 and dt_df[i].month<=9:
        qt_df.append(str(dt_df[i].year)+'Q3')
        #print i,' 3', qt_df
    else:
        qt_df.append(str(dt_df[i].year)+'Q4')
        #print i,' 4', qt_df
    i = i +1
    print i
    #if i > 200:
    #    break
df2=df.assign(Quarter=qt_df)
quarters=df2.groupby('Quarter')
grp_loans_by_credit_pre2009=df2.groupby(['Quarter', 'CreditGrade'])['ListingKey'].count()
grp_loans_by_credit_pre2009.to_csv("grp_loans_by_credit_pre2009.csv")
grp_loans_by_credit_post2009=df2.groupby(['Quarter', 'ProsperRating (Alpha)'])['ListingKey'].count()
grp_loans_by_credit_post2009.to_csv("grp_loans_by_credit_post2009.csv")
grp_loans_by_cat=df2.groupby(['Quarter', 'ListingCategory (numeric)'])['ListingKey'].count()
grp_loans_by_cat.to_csv("grp_loans_by_cat.csv")

grp_prin_by_credit_pre2009=df2.groupby(['Quarter', 'CreditGrade'])['ProsperPrincipalBorrowed'].mean()
grp_prin_by_credit_pre2009.to_csv("grp_prin_by_credit_pre2009.csv")
grp_prin_by_credit_post2009=df2.groupby(['Quarter', 'ProsperRating (Alpha)'])['ProsperPrincipalBorrowed'].mean()
grp_prin_by_credit_post2009.to_csv("grp_prin_by_credit_post2009.csv")
grp_prin_by_cat=df2.groupby(['Quarter', 'ListingCategory (numeric)'])['ProsperPrincipalBorrowed'].mean()
grp_prin_by_cat.to_csv("grp_prin_by_cat.csv")

grp_dti_by_credit_pre2009=df2.groupby(['Quarter', 'CreditGrade'])['DebtToIncomeRatio'].mean()
grp_dti_by_credit_pre2009.to_csv("grp_dti_by_credit_pre2009.csv")
grp_dti_by_credit_post2009=df2.groupby(['Quarter', 'ProsperRating (Alpha)'])['DebtToIncomeRatio'].mean()
grp_dti_by_credit_post2009.to_csv("grp_dti_by_credit_post2009.csv")
grp_dti_by_cat=df2.groupby(['Quarter', 'ListingCategory (numeric)'])['DebtToIncomeRatio'].mean()
grp_dti_by_cat.to_csv("grp_dti_by_cat.csv")

grp_inc_by_credit_pre2009=df2.groupby(['Quarter', 'CreditGrade'])['StatedMonthlyIncome'].mean()
grp_inc_by_credit_pre2009.to_csv("grp_inc_by_credit_pre2009.csv")
grp_inc_by_credit_post2009=df2.groupby(['Quarter', 'ProsperRating (Alpha)'])['StatedMonthlyIncome'].mean()
grp_inc_by_credit_post2009.to_csv("grp_inc_by_credit_post2009.csv")
grp_inc_by_cat=df2.groupby(['Quarter', 'ListingCategory (numeric)'])['StatedMonthlyIncome'].mean()
grp_inc_by_cat.to_csv("grp_inc_by_cat.csv")
#quarter = pd.Timestamp(dt.date(2016, 2, 29)).quarter
