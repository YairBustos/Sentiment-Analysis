# -*- coding: utf-8 -*-
"""
Created on Thu Oct  8 09:49:35 2020

@author: YairBustos
"""

import nltk
nltk.download('vader_lexicon')
#Import VADER Sent Analysis
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from textblob import TextBlob 
import pandas as pd
#Display all columns on Console
pd.set_option('display.max_columns', None)
pd.set_option('display.max_colwidth', None)

file_name = 'raw_dataset'
file_extension = 'csv'
#Create DataFrame with tweets and Sentiment Scores from TextBlob
df = pd.read_csv(r'C:\\YOUR PATH\\' + file_name + '.csv')

#Label tweets using textblob
df['textblob_sent_scores'] = df['text'].apply(lambda text: TextBlob(text).polarity)

#Create a vader instance
vds = SentimentIntensityAnalyzer()
#vader Sent analysis
df['vader_sent_scores'] = df['text'].apply(lambda text: vds.polarity_scores(text))

#Add new columns to the dataset, pos, neg, neutral and compound
df['vader_sent_neg'] = (df['vader_sent_scores'].astype(str)
                                .str.split(",", n=1, expand=True)
                                .loc[:,0]
                                .str.replace("{'neg':","")
                                .str.strip()
                                .astype(float)
                               )
                               
df['vader_sent_neut'] = (df['vader_sent_scores'].astype(str)
                                .str.split(",", n=2, expand=True)
                                .loc[:,1]
                                .str.replace("'neu':","")
                                .str.strip()
                                .astype(float)
                              )
                              
df['vader_sent_pos'] = (df['vader_sent_scores'].astype(str)
                                .str.rsplit(",", n=2, expand=True)
                                .loc[:,1]
                                .str.replace("'pos':","")
                                .str.strip()
                                .astype(float)
                                )
                                
df['vader_sent_comp'] = (df['vader_sent_scores'].astype(str)
                                .str.split(",", n=4, expand=True)
                                .loc[:,3]
                                .str.replace("'compound':","")
                                .str.replace('}','')
                                .str.strip()
                                .astype(float)
                               )

#Create a new column with the tags pos, neg and neut from vader sentiment compound column
df['vader_sent'] = df['vader_sent_comp'].apply(lambda score: 'pos' if score >= 0.05
                                                                    else ('neg' if score <= -0.05 else 'neut'))


#Create a new column with tags pos, neg and neut from TextBlob sentiment
df['TextBlob_sent'] = df['textblob_sent_scores'].apply(lambda score: 'pos' if score >= 0.05
                                                                    else ('neg' if score <= -0.05 else 'neut'))

#Cleaning the DF a bit
cols = ['Unnamed: 0', 'Unnamed: 0.1', 'id', 'date', 'user', 'location', 'retweet_count', 'source', 'user_friends_count', 
        'user_followers_count', 'user_created_at', 'user_listed_count', 'user_favourites_count']

df = df.drop(columns=cols)

#At some point I compared both textblob and vader and I decided to go
#ahead with vader labels

#Remove neutral labeled tweets
df = df[df.vader_sent != 'neut']

print(df['TextBlob_sent'].value_counts())
print(df['vader_sent'].value_counts())
df = df[df.vader_sent != 'neut']
print(df['vader_sent'].value_counts())

#Order by vader scores
print(df.nsmallest(10, ['vader_sent_comp']))         
df.to_csv(r'C:\\YOUR PATH\\' + file_name + '_sentiment' + '.csv', header=True, index=True)





