#In order to compare the behaviour of the ML algorithms, 
#both lemmatization and stemming processes are going to be tested

import pandas as pd
import re
import nltk
from nltk.corpus import stopwords, wordnet
from nltk.tokenize import word_tokenize
from nltk.stem.porter import PorterStemmer
from nltk.stem import WordNetLemmatizer
import contractions
nltk.download('averaged_perceptron_tagger')
nltk.download('wordnet')
pd.set_option('display.max_columns', None)
pd.set_option('display.max_colwidth', None)

#Read CSV file
file_name = 'Raw_dataset'
df = pd.read_csv(r'C:\\YOUR PATH\\' + file_name + '.csv')

#Clean Dataset
def clean(text):
    text = text.lower()                                          #Lowercase Tweets
    text = re.sub(r'https?:\/\/[A-Za-z0-9\.\/]+', '', text)      #Delete URLs
    text = re.sub('@[^\s]+','',text)                             #Remove @Users
    text = text.replace(r"\n", " ")                              #Delete line breaks ("\n")
    text = re.sub(r'\s+',' ', text)                              #Replace two or more spaces with only one
    text = re.sub(r'[^a-zA-z0-9\'\s]', '', text)                 #Remove special characters such as #, -, "
    return text

#Apply clean function to the whole data set
df['text'] = df['text'].apply(clean)

#Expand contractions in column text_no_contractions (returns tokenized text)
df['text_no_contractions'] = df['text'].apply(lambda x: [contractions.fix(word) for word in x.split()])

#Convert tokenized column into strings
df['text_str'] = [' '.join(map(str, each)) for each in df['text_no_contractions']]

#Tokenize again (due to first wrong tokenization process as a result of "contractions" library)
df['tokenized_text'] = df['text_str'].apply(word_tokenize)

#Remove stop words
stop_words = stopwords.words('english')
df['tokenized_text_no_stopwords'] = df['tokenized_text'].apply(lambda x: 
                                                                    [word for word in x 
                                                                     if word not in stop_words])
    
#Stemming 
stemmer = PorterStemmer()
def word_stemmer(text):
    stem_text = ' '.join([stemmer.stem(word) for word in text])
    return stem_text
df['stemmed_text'] = df['tokenized_text_no_stopwords'].apply(lambda x: word_stemmer(x))
df['stemmed_tokenized'] = df['stemmed_text'].apply(word_tokenize)

#Lemmatization
df['pos_tags'] = df['tokenized_text_no_stopwords'].apply(nltk.tag.pos_tag)
print(df['pos_tags'].head())

def get_wordnet_pos(tag):
    if tag.startswith('J'):
        return wordnet.ADJ
    elif tag.startswith('V'):
        return wordnet.VERB
    elif tag.startswith('N'):
        return wordnet.NOUN
    elif tag.startswith('R'):
        return wordnet.ADV
    else:
        return wordnet.NOUN
df['wordnet_pos'] = df['pos_tags'].apply(lambda x: [(word, get_wordnet_pos(pos_tag)) for (word, pos_tag) in x])

WNL = WordNetLemmatizer()
df['lemmatized_tokenized'] = df['wordnet_pos'].apply(lambda x: [WNL.lemmatize(word, tag) for word, tag in x])

#Save to DF
df.to_csv(r'C:\\YOUR PATH\\'+ file_name + '_clean.csv', header=True, index=True)
