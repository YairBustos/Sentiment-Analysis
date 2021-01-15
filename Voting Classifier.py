# -*- coding: utf-8 -*-
"""
Created on Thu Dec 31 10:06:36 2020

@author: YairBustos
"""
from sklearn.metrics import make_scorer, accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_curve, roc_auc_score, auc
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_predict
from sklearn .feature_extraction.text import CountVectorizer
import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib as mlp
import matplotlib.pyplot as plt
from sklearn import model_selection
from sklearn.ensemble import VotingClassifier
from sklearn.svm import LinearSVC
from sklearn.linear_model import SGDClassifier, LogisticRegression

mlp.rcParams.update({'font.size': 22})
pd.set_option('display.max_columns', None)
pd.set_option('display.max_colwidth', None)
pd.set_option('display.max_rows', None)

file_name = 'Dataset_Sentiment_cleaned'
#Import CSV and create DataFrame
df = pd.read_csv(r'C:\\YOUR PATH\\' + file_name + '.csv')

#Using Stemmed Text
#Vectorize using stemmed text
counter = CountVectorizer()    
true_x = counter.fit_transform(df['lemmatized'])

true_y = df['vader_sent'].replace(to_replace=['neg', 'pos'], value=[0, 1])
#print(true_y.head())
kfold = model_selection.KFold(n_splits=10, random_state=42, shuffle=True)

estimators = []
#model1 = LogisticRegression(max_iter=300)
#estimators.append(('lr', model1))
model2 = SGDClassifier()
estimators.append(('sgd', model2))
model3 = LinearSVC(max_iter=2000)
estimators.append(('lsvc', model3))

results = model_selection.cross_val_score(VotingClassifier(estimators), true_x, true_y, cv=kfold)

pred_y_binary = cross_val_predict(VotingClassifier(estimators), true_x, true_y, cv=kfold)
print(pred_y_binary)
"""
pred_y_dec_function = cross_val_predict(VotingClassifier(estimators), true_x, true_y, cv=kfold, method='decision_function')
print(pred_y_dec_function)

proba_y = preprocessing.normalize([pred_y_dec_function])
print(pred_y_proba_normalized)
"""



cf_matrix = confusion_matrix(true_y, pred_y_binary)
print(cf_matrix)
"""
fpr1, tpr1, thresholds1 = roc_curve(true_y, proba_y)
#print(fpr1)
#print(tpr1)
#print(thresholds1)
auc_lr = auc(fpr1, tpr1)
#print(auc_lr)
"""

results_kfold_acc = model_selection.cross_val_score(VotingClassifier(estimators), true_x, true_y, cv=kfold)
print("Accuracy: %.2f%%" % (results_kfold_acc.mean()*100.0)) 
#results_kfold_roc_auc = model_selection.cross_val_score(VotingClassifier(estimators), true_x, true_y, cv=kfold, scoring='roc_auc')
#print("ROC AUC: %.4f" % (results_kfold_roc_auc.mean()*1.0)) 

"""
y_train_pred = cross_val_predict(model_kfold, text_counts, df['vader_sent'], cv=kfold)
cf_matrix = confusion_matrix(df['vader_sent'], y_train_pred)
"""

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
ax.set_title("Voting Classifier (SGD, LinearSVC)")
plt.xlabel('Actual Class', labelpad=10)
plt.ylabel('Predicted Class', labelpad=10)

plt.show()


"""
def plot_roc_curve(fpr, tpr, label=None):
    plt.plot(fpr, tpr, linewidth=2, label=label)
    plt.plot([0,1], [0,1], 'k--')

#plt.plot(fpr1, tpr1, "b:", label="Linear SVC")
plot_roc_curve(fpr1, tpr1, "Voting Classifier (ROC AUC = %.4f" % (auc_lr) + ")")
plt.legend(loc="lower right")
plt.show()
"""