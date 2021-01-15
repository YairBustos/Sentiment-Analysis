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
pd.set_option('display.max_rows', None)

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
#Graph Precision-Recall Curves
fig1, ax1 = plt.subplots(figsize=(10,10))
ax1.plot(rec_lr, pre_lr, label="Logistic Regression (PR AUC = %.4f" % (pr_auc_lr) + ")")
ax1.plot(rec_lsvc, pre_lsvc, label="Linear SVC (PR AUC = %.4f" % (pr_auc_lsvc) + ")")
ax1.plot(rec_sgd, pre_sgd, label="SGD Classifier (PR AUC = %.4f" % (pr_auc_sgd) + ")")
ax1.plot([0, 1], [0.4, 0.4], linestyle='--', label='Baseline')
ax1.set_xlabel('Recall')
ax1.set_ylabel('Precision')
ax1.legend(loc='center left')
plt.show()















