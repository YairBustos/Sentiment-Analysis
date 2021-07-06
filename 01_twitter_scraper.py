
import tweepy as tw
import os
import pandas as pd
import sys

#User login information
consumer_key = 'YOUR DATA'
consumer_secret = 'YOUR DATA'
access_token = 'YOUR DATA'
access_token_secret = 'YOUR DATA'

#Create connection
auth = tw.OAuthHandler(consumer_key, consumer_secret)
auth.set_access_token(access_token, access_token_secret)
api = tw.API(auth, wait_on_rate_limit=True)

#Define search term and date_since as variables
search_words = "COVID19 trump"
date_since = "2020-10-19"
date_until = "2020-10-20"
number_of_tweets = 3000
new_search = search_words + " -filter:retweets"

#Collect Tweets "items" method will return the most recent N numbers_of_tweets
tweets = tw.Cursor(api.search, 
                   q = new_search, 
                   lang = "en", 
                   since = date_since, 
                   until = date_until,
                   tweet_mode='extended').items(number_of_tweets)

#Get a list with tweets' fields
user_locs = [[tweet.id, 
              tweet.created_at, 
              tweet.full_text, 
              tweet.user.screen_name, 
              tweet.user.location, 
              tweet.retweet_count, 
              tweet.source,
              tweet.user.friends_count,
              tweet.user.followers_count,
              tweet.user.created_at,
              tweet.user.listed_count, 
              tweet.user.favourites_count] for tweet in tweets]
#print(user_locs)

#Create a df with the retrieved fields
tweet_df = pd.DataFrame(data = user_locs, columns = ['id', 
                                                    'date', 
                                                    'text', 
                                                    'user', 
                                                    'location', 
                                                    'retweet_count',
                                                    'source',
                                                    'user_friends_count', 
                                                    'user_followers_count', 
                                                    'user_created_at', 
                                                    'user_listed_count', 
                                                    'user_favourites_count'])

#Save df as csv file
tweet_df.to_csv(r'C:\YOUR PATH\Raw_dataset.csv', header=True, index=True)



