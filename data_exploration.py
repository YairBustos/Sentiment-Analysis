# -*- coding: utf-8 -*-
"""
Created on Sat Nov  7 10:15:40 2020

@author: Yair
"""

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.max_colwidth', None)
pd.set_option('float_format', '{:,.2f}'.format)        
plt.rcParams['xtick.labelsize']=16
plt.rcParams['ytick.labelsize']=16

#Import CSV and create DataFrame
file_name = 'Dataset_cleaned'
df = pd.read_csv(r'C:\\YOUR PATH\\' + file_name + '.csv')
#Change datatype to datetime
df['date'] = pd.to_datetime(df['date'])              
#Change datatype to datetime
df['user_created_at'] = pd.to_datetime(df['user_created_at'])               

#Get Timeframe of Data Set
print("Date Column Description")
print(df['date'].describe())
print("\n")

print("Time Frame of Data Set")
print("First Tweet posted on: " + str(df['date'].min()))
print("Last Tweet poster on:  " + str(df['date'].max()))
print("Total Days:            " + str(df['date'].max() - df['date'].min()))


#Get Unique Values from Users
print(df['user'].describe())
print("\n")
print(df['user'].value_counts().head(10))

    
#Location
print(df['location'].describe())
print(df['location'].value_counts().head(10))

locations = df['location'].value_counts()
locations2 = locations.iloc[:9]
locations2.loc['Other'] = locations.iloc[9:].sum()

pie1, ax1 = plt.subplots(figsize=(10, 6), subplot_kw=dict(aspect="equal"))
plt.pie1(locations2.values, labels=locations2.index, autopct="%1.1f%%", 
        explode=[0.03]*10, pctdistance=0.8, startangle=130)
plt.title("User Locations", fontsize=14)

plt.show()
plt.close()

#Retweet Count Column Description
print("Retweet Count Descriptive Statistics")
print(df['retweet_count'].describe())


#Source column Description
print("Source Column Information")
print("Diferent Values")
print(df['source'].describe())

sources = df['source'].value_counts()
sources2 = sources.iloc[:4]
sources2.loc['Other'] = sources.iloc[4:].sum()


pie2, ax2 = plt.subplots(figsize=[10,6])
plt.pie2(sources2.values, labels=sources2.index, autopct="%.1f%%", 
        explode=[0.03]*5, pctdistance=0.8)
plt.title("Tweet Sources", fontsize=14)
plt.show()
plt.close()

df2 = df[df['user_friends_count'] <= 5000]
print(df2.shape)
#print(df2.sort_values(by=['user_friends_count'], ascending=False).head(50))


#User Friends Count
print("User Friends Count Descriptive Statistics")
print(df['user_friends_count'].describe())

fig3, ax3 = plt.subplots(figsize=[10,6])
plt.hist(df2['user_friends_count'], bins=1000)
plt.title("User Friends Count Histogram", fontsize=25)
plt.xlabel("User Friends Count", fontsize=20)
plt.ylabel("Frequency", fontsize=20)
plt.show()
plt.close()


df3 = df[df['user_followers_count'] <= 2000]
print(df3.shape)
#print(df2.sort_values(by=['user_friends_count'], ascending=False).head(50))
#User Followers Count
print("User Followers Count Descriptive Statistics")
print(df['user_followers_count'].describe())


fig4, ax4 = plt.subplots(figsize=[10,6])
plt.hist(df3['user_followers_count'], bins=1000)
plt.title("User Followers Count Histogram", fontsize=25)
plt.xlabel("User Followers Count", fontsize=20)
plt.ylabel("Frequency", fontsize=20)
plt.show()
plt.close()

#User Created At Column Description
print("User Created at Descriptive Statistics")
print("Oldest User on Data Set: " + str(df['user_created_at'].min()))
print("Newest User on Data Set:  " + str(df['user_created_at'].max()))
print("Mean of User Created at: " + str(df['user_created_at'].mean()))



#User Listed Count Column Description
print("User Listed Count Descriptive Statistics")
print(df['user_listed_count'].describe())
df4 = df[df['user_listed_count'] <= 100]
print(df4.shape)

fig5, ax5 = plt.subplots(figsize=[10,6])
plt.hist(df4['user_listed_count'], bins=200)
plt.title("User Listed Count Histogram", fontsize=25)
plt.xlabel("User Listed Count", fontsize=20)
plt.ylabel("Frequency", fontsize=20)
plt.show()
plt.close()


#User Favourites Count
print("User Favourites Count Descriptive Statistics")
print(df['user_favourites_count'].describe())

df5 = df[df['user_favourites_count'] <= 10000]

fig6, ax6 = plt.subplots(figsize=[10,6])
plt.hist(df5['user_favourites_count'], bins=1000)
plt.title("User Favourites Count Histogram", fontsize=25)
plt.xlabel("User Favourites Count", fontsize=20)
plt.ylabel("Frequency", fontsize=20)
ax6.ticklabel_format(style='plain', axis='x')
plt.show()
plt.close()


































