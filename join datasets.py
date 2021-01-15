# -*- coding: utf-8 -*-
"""
Created on Tue Oct 20 11:10:36 2020

@author: YairBustos
"""

import pandas as pd

pd.set_option('display.max_columns', None)
#pd.set_option('display.max_colwidth', None)


#Create a dataframe by sampling all the tweet datasets
rows_per_dataframe = 1000                           #Number of rows to sample
df = pd.DataFrame()                                 #Create empty dataframe
for day in range(2, 20):
    file_name = 'Tweets_' + str(day) + '_Oct_2020_3k_COVID19_trump'         #Create name of csv           
    #Import CSV and create DataFrame
    df0 = pd.read_csv(r'C:\\YOUR PATH\\' + file_name + '.csv')  #Read csv file
    df0 = df0.sample(n=rows_per_dataframe)                                  #Sample csv
    df = df.append(df0)                                                     #Append sample to empty dataframe

df.reset_index(inplace=True)                                                #Reset index of df
#print(df.head())   
#print(df.tail()) 
print(df.shape)                                                             #Print shape to make sure it worked

df.to_csv(r'C:\\YOUR PATH\\Raw_dataset.csv', header=True, index=True)  #Export df to csv file