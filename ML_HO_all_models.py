# -*- coding: utf-8 -*-
"""
Created on Wed Oct 21 13:17:46 2020

@author: YairBustos
"""
import pandas as pd
import numpy as np
from sklearn import metrics
from sklearn.metrics import f1_score, precision_score, recall_score
from sklearn.naive_bayes import MultinomialNB
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer, HashingVectorizer

pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.max_colwidth', None)

file_name = 'dataset_cleaned'
#Import CSV and create DataFrame
df = pd.read_csv(r'C:\\YOUR PATH\\' + file_name + '.csv')

#Models to be used
models = {'Naive Bayes': MultinomialNB(),
          'Decision Tree': tree.DecisionTreeClassifier(),
          'Random Forest': RandomForestClassifier(n_estimators=50),
          'Linear SVC': LinearSVC(max_iter=2000),
          'Logistic Regression': LogisticRegression(max_iter=300),
          'KNN': KNeighborsClassifier(),
          'SGD': SGDClassifier(),
          'GTB': GradientBoostingClassifier()}   

#Vectorize using stemmed text
texts = {'Stemmed Text': df['stemmed_text'], 
         'Lemmatized Text': df['lemmatized']}

vectors = {'Tf-idfVectorizer': TfidfVectorizer(),
           'CountVectorizer': CountVectorizer()}

df_perf = pd.DataFrame(columns=['Text',
                                'Vector',
                                'Model',
                                'Splitting',
                                'Accuracy', 
                                'F1 Score',
                                'Precision', 
                                'Recall'])
t_sizes = {'Holdout 80/20': 0.2,'Holdout 75/25':  0.25,'Holdout 70/30':  0.3}

for t_size in t_sizes:
    print(t_size)

    for text in texts:
        print(text)
        
        for vector in vectors:
            count = vectors[vector]
            print(count)
            text_counts = count.fit_transform(texts[text]).toarray()     
        
            #Iterate trough models
            for model in models:
                print(model)
                #Split Dataframe into train and test
                X_train, X_test, Y_train, Y_test = train_test_split(text_counts, df['vader_sent'],          
                                                                test_size=t_sizes[t_size], random_state=42, stratify = df['vader_sent'])
                #Create a model from dictionary of models
                classifier = models[model]
                #Fit model
                classifier.fit(X_train, Y_train)
                #Prediction
                predicted = classifier.predict(X_test)
                #Print Separator for each model
                #print("\n" + model + ", " + text + ", Test Size: " + str(t_size) +" Performance Indicators")
                #Get Accuracy
                accuracy_score = metrics.accuracy_score(predicted, Y_test)
                print(text + " | " + vector + " | " + model + " | " + str(t_size) + " | Accuracy: " + '{:04.2f}'.format(accuracy_score*100)+'%')
                #Get F1 Score
                f1score_pos = f1_score(predicted, Y_test, pos_label="pos")
                print(text + " | " + vector + " | " + model + " | " + str(t_size) + " | F1 Score (pos): " + '{:04.2f}'.format(f1score_pos*100)+'%')
                f1score_neg = f1_score(predicted, Y_test, pos_label="neg")
                print(text + " | " + vector + " | " + model + " | " + str(t_size) + " | F1 Score (neg): " + '{:04.2f}'.format(f1score_neg*100)+'%')
                #Get Precision
                precision_pos = precision_score(predicted, Y_test, pos_label="pos")
                print(text + " | " + vector +  " | " + model + " | " + str(t_size) + " | Precision (pos): " + '{:04.2f}'.format(precision_pos*100)+'%')    
                precision_neg = precision_score(predicted, Y_test, pos_label="neg")
                print(text + " | " + vector +  " | " + model + " | " + str(t_size) + " | Precision (neg): " + '{:04.2f}'.format(precision_neg*100)+'%')
                #Get Recall
                recall_pos = recall_score(predicted, Y_test, pos_label="pos")
                print(text + " | " + vector +  " | " + model + " | " + str(t_size) + " | Recall (pos): " + '{:04.2f}'.format(recall_pos*100)+'%')    
                recall_neg = recall_score(predicted, Y_test, pos_label="neg")
                print(text + " | " + vector +  " | " + model + " | " + str(t_size) + " | Recall (neg): " + '{:04.2f}'.format(recall_neg*100)+'%')
                print("")
                #Add performances to dataframe
                perf_list = [{'Text': text,
                              'Vector': vector,
                              'Model': model,
                              'Splitting': t_size,
                              'Accuracy': '{:04.2f}'.format(accuracy_score*100)+'%', 
                              'F1 Score': '{:04.2f}'.format(f1score_neg*100)+'%', 
                              'Precision': '{:04.2f}'.format(precision_neg*100)+'%', 
                              'Recall': '{:04.2f}'.format(recall_neg*100)+'%'}]
                df_perf = df_perf.append(perf_list, ignore_index = True, sort=False)

print(df_perf.head())
df_perf.to_csv(r'C:\\YOUR PATH\\HO_performances.csv', header=True, index=True)








