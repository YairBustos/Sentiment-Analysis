#Since this is a sentiment analysis project and the scrapped tweets are not labeled, 
#it is necessary to label them, either manually (which was discarded due to high costs)
#or automatically, using two very well known libraries for labeling, VADER and TEXTBLOB
#at the end, a comparison was made and vader was chosen as the label to keep

import nltk
nltk.download('vader_lexicon')
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from textblob import TextBlob 
import pandas as pd
#Display all columns on Console
pd.set_option('display.max_columns', None)
pd.set_option('display.max_colwidth', None)

file_name = 'raw_dataset'
file_extension = 'csv'

#Create DataFrame from CSV file
df = pd.read_csv(r'C:\\YOUR PATH\\' + file_name + '.csv')


######## TEXTBLOB #########
#Textblob is very straight forward, it gives a score in the range of [-1, 1]
#being -1 the most negative sentiment and +1 the most positive sentiment
#Label tweets using textblob
df['textblob_sent_scores'] = df['text'].apply(lambda text: TextBlob(text).polarity)





######## VADER #########
#Vader on the other hand gives a dictionary of 4 different scores: positive, negative, 
#neutral and compound (which is computed using the first three scores)

#Create a vader instance
vds = SentimentIntensityAnalyzer()
#vader Sent analysis (column with the before mentioned dictionary)
df['vader_sent_scores'] = df['text'].apply(lambda text: vds.polarity_scores(text))

#Add new column with vader NEGATIVE score
df['vader_sent_neg'] = (df['vader_sent_scores'].astype(str)                 #convert it to str type
                                .str.split(",", n=1, expand=True)           #split it where the comma is (it's a dictionary)
                                .loc[:,0]                                   #locate first element
                                .str.replace("{'neg':","")                  #replace letters by "" (we want to get only the number (score))
                                .str.strip()                                #strip what's needed
                                .astype(float)                              #convert it to float
                               )

#Add new column with vader NEUTRAL score
df['vader_sent_neut'] = (df['vader_sent_scores'].astype(str)
                                .str.split(",", n=2, expand=True)
                                .loc[:,1]
                                .str.replace("'neu':","")
                                .str.strip()
                                .astype(float)
                              )

#Add new column with vader POSITIVE score
df['vader_sent_pos'] = (df['vader_sent_scores'].astype(str)
                                .str.rsplit(",", n=2, expand=True)
                                .loc[:,2]
                                .str.replace("'pos':","")
                                .str.strip()
                                .astype(float)
                                )

#Add new column with vader COMPOUND score                                
df['vader_sent_comp'] = (df['vader_sent_scores'].astype(str)
                                .str.split(",", n=4, expand=True)
                                .loc[:,3]
                                .str.replace("'compound':","")
                                .str.replace('}','')
                                .str.strip()
                                .astype(float)
                               )




#Create a new column with the tags "pos", "neg" and "neut" from VADER sentiment compound score
#IF
#COMPOUND score <= -0.5  == "neg" (negative)
# -0.05 < COMPOUND SCORE < +0.05  ==  "neut" (neutral)
#COMPOUND score >= 0.5  == "pos" (positive)
df['vader_sent'] = df['vader_sent_comp'].apply(lambda score: 'pos' if score >= 0.05
                                                                    else ('neg' if score <= -0.05 else 'neut'))


#Create a new column with tags "pos", "neg" and "neut" from TextBlob sentiment score
#Same logic as VADER
df['TextBlob_sent'] = df['textblob_sent_scores'].apply(lambda score: 'pos' if score >= 0.05
                                                                    else ('neg' if score <= -0.05 else 'neut'))

#Cleaning the DF a bit
cols = ['Unnamed: 0', 'Unnamed: 0.1', 'id', 'date', 'user', 'location', 'retweet_count', 'source', 'user_friends_count', 
        'user_followers_count', 'user_created_at', 'user_listed_count', 'user_favourites_count']

df = df.drop(columns=cols)

#At some point I compared both textblob and vader and I decided to go
#ahead with vader labels

#In order to have a binomial variable for a better ML process,
#neutral labeled tweets are removed
df = df[df.vader_sent != 'neut']

print(df['TextBlob_sent'].value_counts())
print(df['vader_sent'].value_counts())
df = df[df.vader_sent != 'neut']
print(df['vader_sent'].value_counts())

#Order by vader scores
print(df.nsmallest(10, ['vader_sent_comp']))         
df.to_csv(r'C:\\YOUR PATH\\' + file_name + '_sentiment' + '.csv', header=True, index=True)





