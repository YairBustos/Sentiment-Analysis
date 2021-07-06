
#Confusion Matrix Logistic Regression
#Lemmatized Text
#BoW
from sklearn.metrics import make_scorer, accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_predict
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn .feature_extraction.text import CountVectorizer
import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib as mlp
import matplotlib.pyplot as plt
from sklearn import model_selection
pd.set_option('display.max_rows', None)
mlp.rcParams.update({'font.size': 22})
file_name = 'Dataset_Sentiment_cleaned'
#Import CSV and create DataFrame
df = pd.read_csv('file_name + '.csv')

#Using Stemmed Text
#Vectorize using stemmed text
counter = CountVectorizer()    
text_counts = counter.fit_transform(df['lemmatized'])

x1 = text_counts
y1 = df['vader_sent']

kfold = model_selection.KFold(n_splits=10, random_state=42, shuffle = True)
model_kfold = LogisticRegression(max_iter=300)

results_kfold = model_selection.cross_val_score(model_kfold, x1, y1, cv=kfold)

#results_kfold2 = model_selection.cross_val_score(model_kfold, x1, y1, cv=kfold, scoring=make_scorer(f1_score, average='micro'))
results_kfold3 = model_selection.cross_val_score(model_kfold, x1, y1, cv=kfold, scoring='f1_macro')
#results_kfold4 = model_selection.cross_val_score(model_kfold, x1, y1, cv=kfold, scoring='recall_micro')
results_kfold5 = model_selection.cross_val_score(model_kfold, x1, y1, cv=kfold, scoring='recall_macro')
#results_kfold6 = model_selection.cross_val_score(model_kfold, x1, y1, cv=kfold, scoring='precision_micro')
results_kfold7 = model_selection.cross_val_score(model_kfold, x1, y1, cv=kfold, scoring='precision_macro')
print("Accuracy: %.2f%%" % (results_kfold.mean()*100.0)) 
#print("F1 Micro: %.2f%%" % (results_kfold2.mean()*100.0)) 
print("F1 Macro: %.2f%%" % (results_kfold3.mean()*100.0)) 
#print("Recall Micro: %.2f%%" % (results_kfold4.mean()*100.0)) 
print("Recall Macro: %.2f%%" % (results_kfold5.mean()*100.0)) 
#print("Precision Micro: %.2f%%" % (results_kfold6.mean()*100.0)) 
print("Precision Macro: %.2f%%" % (results_kfold7.mean()*100.0)) 

y_train_pred = cross_val_predict(model_kfold, text_counts, df['vader_sent'], cv=kfold)
cf_matrix = confusion_matrix(df['vader_sent'], y_train_pred)
print(cf_matrix)
group_names = ['True Negative','False Positive','False Negative','True Positive']
group_counts = ['{0:0.0f}'.format(value) for value in
                cf_matrix.flatten()]
group_percentages = ['{0:.2%}'.format(value) for value in
                     cf_matrix.flatten()/np.sum(cf_matrix)]
labels = [f'{v1}\n{v2}\n{v3}' for v1, v2, v3 in
          zip(group_names,group_counts,group_percentages)]
labels = np.asarray(labels).reshape(2,2)

fig, ax = plt.subplots(figsize=(10,10))
ax = sns.heatmap(cf_matrix, annot=labels, fmt='', cmap='Blues',
                 vmin=0, vmax=25000)

ylabel = ['Negative', 'Positive']
ax.set_xticklabels(ylabel, ha='center')
ax.set_yticklabels(ylabel, va='center')
ax.set_title("Confusion Matrix Linear Regression")
plt.xlabel('Actual Class', labelpad=10)
plt.ylabel('Predicted Class', labelpad=10)

plt.show()
