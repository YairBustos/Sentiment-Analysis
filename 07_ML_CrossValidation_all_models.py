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
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn import model_selection
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.max_colwidth', None)

file_name = 'clean_dataset'
#Import CSV and create DataFrame
df = pd.read_csv(file_name + '.csv')

#Models to be used
models = {'Naive Bayes': MultinomialNB(),
          'Decision Tree': tree.DecisionTreeClassifier(),
          'Random Forest': RandomForestClassifier(n_estimators=50),
          'Linear SVC': LinearSVC(max_iter=2000),
          'Logistic Regression': LogisticRegression(max_iter=300),
          'KNN': KNeighborsClassifier(),
          'SGD': SGDClassifier(),
          'GTB': GradientBoostingClassifier(n_estimators=50)}   

#Vectorize using stemmed text
texts = {'Stemmed Text': df['stemmed_tokenized'], 
         'Lemmatized Text': df['lemmatized_tokenized']}

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

kfolds_list = {'CV 5 folds': 5,
               'CV 10 folds': 10}

for k in kfolds_list:
    print(k)

    for text in texts:
        print(text)
        
        for vector in vectors:
            count = vectors[vector]
            print(count)
            text_counts = count.fit_transform(texts[text])
        
            #Iterate through models
            for model in models:
                print(model)
                #Create a model from dictionary of models
                kfold = model_selection.KFold(n_splits=kfolds_list[k], random_state=42, shuffle = True)
                model_kfold = models[model]
                
                #Get Accuracy
                results_kfold_accuracy = model_selection.cross_val_score(model_kfold, text_counts, df['vader_sent'], cv=kfold)
                #print("Accuracy: %.2f%%" % (results_kfold_accuracy.mean()*100.0)) 
                print(text + " | " + vector + " | " + model + " | " + str(k) + " | Accuracy: " + '{:04.2f}'.format(results_kfold_accuracy.mean()*100)+'%')
                #Get F1 Score
                results_kfold_f1 = model_selection.cross_val_score(model_kfold, text_counts, df['vader_sent'], cv=kfold, scoring='f1_macro')
                print(text + " | " + vector + " | " + model + " | " + str(k) + " | F1 Score: " + '{:04.2f}'.format(results_kfold_f1.mean()*100)+'%')
                #Get Precision
                results_kfold_recall = model_selection.cross_val_score(model_kfold, text_counts, df['vader_sent'], cv=kfold, scoring='recall_macro')
                print(text + " | " + vector +  " | " + model + " | " + str(k) + " | Precision: " + '{:04.2f}'.format(results_kfold_recall.mean()*100)+'%')    
                #Get Recall
                results_kfold_precision = model_selection.cross_val_score(model_kfold, text_counts, df['vader_sent'], cv=kfold, scoring='precision_macro')
                print(text + " | " + vector +  " | " + model + " | " + str(k) + " | Recall: " + '{:04.2f}'.format(results_kfold_precision.mean()*100)+'%')    
                print("")
                
                #Add performances to dataframe
                perf_list = [{'Text': text,
                              'Vector': vector,
                              'Model': model,
                              'Splitting': k,
                              'Accuracy': '{:04.2f}'.format(results_kfold_accuracy.mean()*100)+'%',
                              'F1 Score': '{:04.2f}'.format(results_kfold_f1.mean()*100)+'%', 
                              'Precision': '{:04.2f}'.format(results_kfold_recall.mean()*100)+'%', 
                              'Recall': '{:04.2f}'.format(results_kfold_precision.mean()*100)+'%'}]
                df_perf = df_perf.append(perf_list, ignore_index = True, sort=False)

print(df_perf)
df_perf.to_csv('CV_performances.csv', header=True, index=True)

