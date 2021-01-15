# -*- coding: utf-8 -*-
"""
Created on Tue Dec 22 22:02:11 2020

@author: yairb
"""
from sklearn.svm import LinearSVC
from sklearn.linear_model import SGDClassifier, LogisticRegression
from sklearn.metrics import make_scorer, accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_curve, roc_auc_score, auc, precision_recall_curve
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_predict
from sklearn .feature_extraction.text import CountVectorizer
import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib as mlp
import matplotlib.pyplot as plt
from sklearn import model_selection
from sklearn import preprocessing
import re
import nltk
import keras
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras import metrics
import tensorflow as tf
import contractions
nltk.download('punkt')
nltk.download('stopwords')

mlp.rcParams.update({'font.size': 22})
pd.set_option('display.max_columns', None)
pd.set_option('display.max_colwidth', None)
pd.set_option('display.max_rows', None)
#########################
#Import CSV and create DataFrame
file_name = 'Dataset_Sentiment_cleaned'
df = pd.read_csv(r'C:\\YOUR PATH\\' + file_name + '.csv')
#########################
#Create Vector BoW using CountVectorizer
counter = CountVectorizer()    
true_x = counter.fit_transform(df['lemmatized'])
#########################
#Get labels and transform them to int (from 'neg' and 'pos' to 0 and 1)
true_y = df['vader_sent'].replace(to_replace=['neg', 'pos'], value=[0, 1])
#########################
#Create CV
kfold = model_selection.KFold(n_splits=10, random_state=42, shuffle=True)
#########################
#Create Classifiers
sgd_clf = SGDClassifier()
lr_clf = LogisticRegression(max_iter=300)
lsvc_clf = LinearSVC(max_iter=2000)
#########################
#Create binary predictions
print("Binary Prediction")
#Predictions of Logistic Regression
pred_y_binary_lr = cross_val_predict(lr_clf, true_x, true_y, cv=kfold)
print(pred_y_binary_lr)
#Predictions of SGD Classifier
pred_y_binary_sgd = cross_val_predict(sgd_clf, true_x, true_y, cv=kfold)
print(pred_y_binary_sgd)
#Predictions of LinearSVC
pred_y_binary_lsvc = cross_val_predict(lsvc_clf, true_x, true_y, cv=kfold)
print(pred_y_binary_lsvc)
#########################
#Prediction Probabilities
print("Prediction Probabilities")
#Prediction probabilities Logistic Regression
pred_y_lr = cross_val_predict(lr_clf, true_x, true_y, cv=kfold, method='predict_proba')
print(pred_y_lr)
proba_y_lr = pred_y_lr[:, 1]
print(proba_y_lr)

#Prediction "probabilities" Linear SVC (Get decision function and normalize it)
pred_y_dec_function_lsvc = cross_val_predict(lsvc_clf, true_x, true_y, cv=kfold, method='decision_function')
print(pred_y_dec_function_lsvc)
pred_y_proba_normalized_lsvc = preprocessing.normalize([pred_y_dec_function_lsvc])
print(pred_y_proba_normalized_lsvc)

#Prediction "probabilities" SGD Classifier (Get decision function and normalize it)
pred_y_dec_function_sgd = cross_val_predict(sgd_clf, true_x, true_y, cv=kfold, method='decision_function')
print(pred_y_dec_function_sgd)
pred_y_proba_normalized_sgd = preprocessing.normalize([pred_y_dec_function_sgd])
print(pred_y_proba_normalized_sgd)
#########################

#Confusion Matrix
print("Confusion Matrices")
#Confusion Matrix Logistic Regression
cf_matrix_lr = confusion_matrix(true_y, pred_y_binary_lr)
print(cf_matrix_lr)

#Confusion Matrix Linear SVC
cf_matrix_lsvc = confusion_matrix(true_y, pred_y_binary_lsvc)
print(cf_matrix_lsvc)

#Confusion Matrix SGD Classifier
cf_matrix_sgd = confusion_matrix(true_y, pred_y_binary_sgd)
print(cf_matrix_sgd)
#########################

#Get False Positive Rate, True Positive Rate and Thresholds of each model
fpr_lr, tpr_lr, thresholds_lr = roc_curve(true_y, proba_y_lr)
fpr_lsvc, tpr_lsvc, thresholds_lsvc = roc_curve(true_y, pred_y_proba_normalized_lsvc[0])
fpr_sgd, tpr_sgd, thresholds_sgd = roc_curve(true_y, pred_y_proba_normalized_sgd[0])

#Get AUC of each model
roc_auc_lr = auc(fpr_lr, tpr_lr)
roc_auc_lsvc = auc(fpr_lsvc, tpr_lsvc)
roc_auc_sgd = auc(fpr_sgd, tpr_sgd)
print(roc_auc_lr)
print(roc_auc_lsvc)
print(roc_auc_sgd)
#########################
"""
#Graph ROC Curves
fig, ax = plt.subplots(figsize=(10,10))
ax.plot(fpr_lr, tpr_lr, label="Logistic Regression (ROC AUC = %.4f" % (roc_auc_lr) + ")")
ax.plot(fpr_lsvc, tpr_lsvc, label="Linear SVC (ROC AUC = %.4f" % (roc_auc_lsvc) + ")")
ax.plot(fpr_sgd, tpr_sgd, label="SGD Classifier (ROC AUC = %.4f" % (roc_auc_sgd) + ")")
plt.plot([0,1], [0,1], 'k--')
ax.set_xlabel('Recall')
ax.set_ylabel('Precision')
ax.legend(loc='lower right')
plt.show()
plt.close()
"""

#########################
#Get Precision, Recall and Thresholds of each model
pre_lr, rec_lr, thresholds2_lr = precision_recall_curve(true_y, proba_y_lr)
pre_lsvc, rec_lsvc, thresholds2_lsvc = precision_recall_curve(true_y, pred_y_proba_normalized_lsvc[0])
pre_sgd, rec_sgd, thresholds2_sgd = precision_recall_curve(true_y, pred_y_proba_normalized_sgd[0])

#Get AUC of each model
pr_auc_lr = auc(rec_lr, pre_lr)
pr_auc_lsvc = auc(rec_lsvc, pre_lsvc)
pr_auc_sgd = auc(rec_sgd, pre_sgd)
print(pr_auc_lr)
print(pr_auc_lsvc)
print(pr_auc_sgd)

#########################
"""
#Graph Precision-Recall Curves
fig1, ax1 = plt.subplots(figsize=(10,10))
ax1.plot(rec_lr, pre_lr, label="Logistic Regression (PR AUC = %.4f" % (roc_auc_lr) + ")")
ax1.plot(rec_lsvc, pre_lsvc, label="Linear SVC (PR AUC = %.4f" % (roc_auc_lsvc) + ")")
ax1.plot(rec_sgd, pre_sgd, label="SGD Classifier (PR AUC = %.4f" % (roc_auc_sgd) + ")")
ax1.plot([0, 1], [0.4, 0.4], linestyle='--', label='Baseline')
ax1.set_xlabel('Recall')
ax1.set_ylabel('Precision')
ax1.legend(loc='center left')
plt.show()
plt.close()
"""


##########################################
#NN Part
#Import CSV and create DataFrame
file_name2 = '2900_per_day_sentiment'
#Create DataFrame with tweets and Sentiment Scores from TextBlob
df2 = pd.read_csv(r'C:\\Users\\Yairb\\Desktop\\PythonPP\\Datasets\\' + file_name2 + '.csv')

#########################################
#Clean Text
def clean(text):
    text = text.lower()                                          #Lowercase Tweets
    text = re.sub(r'https?:\/\/[A-Za-z0-9\.\/]+', '', text)      #Delete URLs
    text = re.sub('@[^\s]+','',text)                             #Remove @Users
    text = text.replace(r"\n", " ")                              #Delete "\n"
    text = re.sub(r'\s+',' ', text)                              #Replace two or more spaces with only one
    #Remove special characters such as #, -, " (? and ! in)
    #Hashtags are not removed, only the hash symbol
    text = re.sub(r'[^a-zA-z!?\'\s]', '', text)

    return text

#Apply cleaning function to text
df2['text'] = df2['text'].apply(clean)
df2 = df2.drop(columns=['textblob_sent_scores', 'vader_sent_comp', 'TextBlob_sent'])
#Remove "neutral" sentiment labeled rows
df2 = df2[df.vader_sent != 'neut']


#Expand contractions tokenizes but expanded contraction are in the same item inside the list
#e.g.: [there is, a, cat] -> "there is" is what has to be avoided
df2['text_no_contractions'] = df2['text'].apply(lambda x: [contractions.fix(word) for word in x.split()])

#Conver to str
df2['text_str'] = [' '.join(map(str, each)) for each in df2['text_no_contractions']]

#Expand contractions that where not expanded before
df2['text_str'] = df2['text_str'].str.replace('\'s',' is')
df2['text_str'] = df2['text_str'].str.replace('\'','')            #Replace the rest of quotes for ' '


#Tokenize again
df2['tokenized_text'] = df2['text_str'].apply(word_tokenize)

#Remove stop words
stop_words = stopwords.words('english')
df2['tokenized_no_stopwords'] = df2['tokenized_text'].apply(lambda x: 
                                                                    [word for word in x 
                                                                     if word not in stop_words])

df2['str_no_stopwords'] = [' '.join(map(str, each)) for each in df2['tokenized_no_stopwords']]

#Pad Texts for NN
tweet = df2['str_no_stopwords'].values
tokenizer = Tokenizer(num_words=5000)
tokenizer.fit_on_texts(tweet)
vocab_size = len(tokenizer.word_index) + 1
encoded_docs = tokenizer.texts_to_sequences(tweet)
padded_sequence = pad_sequences(encoded_docs, maxlen=200)

print("Padded Sequence: ", padded_sequence)
print("Padded Sequence Length: ", len(padded_sequence))

#Convert "neg" and "pos" to 0 and 1
sentiment_label = df2['vader_sent'].replace(to_replace=['neg', 'pos'], value=[0, 1])
print("Sentiment Label: ", sentiment_label)
print("Sentiment Label Length: ", len(sentiment_label))

from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

#Split data with 0.2 test
X_train, X_test, y_train, y_test = train_test_split(padded_sequence, 
                                                    sentiment_label, 
                                                    test_size=0.2)

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.layers import SpatialDropout1D
from tensorflow.keras.layers import Embedding

#Create function to generate LSTM model
embedding_vector_length = 32
def build_model(vocab_size, embedding_vector_length):
    model = Sequential()
    model.add(Embedding(vocab_size, embedding_vector_length, input_length=200))
    model.add(SpatialDropout1D(0.25))
    model.add(LSTM(50, dropout=0.5, recurrent_dropout=0.5))
    model.add(Dropout(0.2))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy',optimizer='adam', metrics=['accuracy',
                                                                        tf.keras.metrics.Precision(),
                                                                        tf.keras.metrics.Recall(), 
                                                                        tf.keras.metrics.AUC(), 
                                                                        tf.keras.metrics.FalseNegatives(),
                                                                        tf.keras.metrics.FalsePositives(), 
                                                                        tf.keras.metrics.TrueNegatives(), 
                                                                        tf.keras.metrics.TruePositives()])
    model.summary()
    return model

#Build model and fit
from keras.wrappers.scikit_learn import KerasClassifier
keras_model = build_model(vocab_size, embedding_vector_length)
keras_model.fit(X_train, y_train, epochs=5, batch_size=100, verbose=1)

#Predict labels using X test
from sklearn.metrics import roc_curve
y_pred_keras = keras_model.predict(X_test).ravel()
print("Y Pred Keras: ", y_pred_keras)
#Get False Positive Rate, True Positive Rate and Thresholds
fpr_keras, tpr_keras, thresholds_keras = roc_curve(y_test, y_pred_keras)
print("FPR: ", fpr_keras)
print("TPR: ", tpr_keras)

from sklearn.metrics import auc
roc_auc_keras = auc(fpr_keras, tpr_keras)
######################################
#Plot ROC Curve
#Graph ROC Curves
fig, ax = plt.subplots(figsize=(15,15))
ax.plot(fpr_lr, tpr_lr, label="Logistic Regression (ROC AUC = %.4f" % (roc_auc_lr) + ")")
ax.plot(fpr_lsvc, tpr_lsvc, label="Linear SVC (ROC AUC = %.4f" % (roc_auc_lsvc) + ")")
ax.plot(fpr_sgd, tpr_sgd, label="SGD Classifier (ROC AUC = %.4f" % (roc_auc_sgd) + ")")
ax.plot(fpr_keras, tpr_keras, label="LSTM (ROC AUC = %.4f" % (roc_auc_keras) + ")")
plt.plot([0,1], [0,1], 'k--')
plt.title('ROC Curve')
ax.set_xlabel('False positive rate')
ax.set_ylabel('True positive rate')
ax.legend(loc='lower right')
plt.show()
plt.close()

######################################
pre_lstm, rec_lstm, thresholds2_lstm = precision_recall_curve(y_test, y_pred_keras)
pr_auc_lstm = auc(rec_lstm, pre_lstm)
#Graph Precision-Recall Curves
fig1, ax1 = plt.subplots(figsize=(15,15))
ax1 = sns.lineplot()
ax1.plot(rec_lr, pre_lr, label="Logistic Regression (PR AUC = %.4f" % (pr_auc_lr) + ")")
ax1.plot(rec_lsvc, pre_lsvc, label="Linear SVC (PR AUC = %.4f" % (pr_auc_lsvc) + ")")
ax1.plot(rec_sgd, pre_sgd, label="SGD Classifier (PR AUC = %.4f" % (pr_auc_sgd) + ")")
ax1.plot(rec_sgd, pre_sgd, label="LSTM (PR AUC = %.4f" % (pr_auc_lstm) + ")")
ax1.plot([0, 1], [0.4, 0.4], linestyle='--', label='Baseline')
ax1.set_xlabel('Recall')
ax1.set_ylabel('Precision')
ax1.legend(loc='center left')
plt.show()
plt.close()

#print("y test: ", y_test)
#print("y_pred_keras: ", y_pred_keras)
#print("y_pred_keras.round(): ", y_pred_keras.round())


cf_matrix = confusion_matrix(y_test, y_pred_keras.round())
#cf_matrix = [[24973, (4476+1583+1196+1044+973)], 
#             [(948+1404+1116+943+889), (9636+9180+9468+9641+9695)]]
group_names = ['True Negative','False Positive','False Negative','True Positive']
group_counts = ['{0:0.0f}'.format(value) for value in
                cf_matrix.flatten()]
group_percentages = ['{0:.2%}'.format(value) for value in
                     cf_matrix.flatten()/np.sum(cf_matrix)]
labels = [f'{v1}\n{v2}\n{v3}' for v1, v2, v3 in
          zip(group_names,group_counts,group_percentages)]
labels = np.asarray(labels).reshape(2,2)

fig2, ax2 = plt.subplots(figsize=(10,10))
ax2 = sns.heatmap(cf_matrix, annot=labels, fmt='', cmap='Blues',
                 vmin=0, vmax=5000)
ylabel = ['Negative', 'Positive']
ax2.set_xticklabels(ylabel, ha='center')
ax2.set_yticklabels(ylabel, va='center')
ax2.set_title("Confusion Matrix LSTM")
plt.xlabel('Actual Class', labelpad=10)
plt.ylabel('Predicted Class', labelpad=10)
plt.show()






