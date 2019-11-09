# -*- coding: utf-8 -*-
"""
Created on Tue Jul 16 19:00:16 2019

@author: Aashita
"""

import pandas as pd, numpy as np, re, time
from nltk.stem.porter import PorterStemmer
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.svm import LinearSVC
from sklearn.model_selection import cross_val_score
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

# Loading data from json file
data = pd.read_json("C:/Users/HP/Downloads/Sarcasm_Headlines_Dataset.json", lines = True)

print(data.isnull().any(axis = 0))

data['headline'] = data['headline'].apply(lambda s : re.sub('[^a-zA-Z]', ' ', s))

# getting features and labels
features = data['headline']
labels = data['is_sarcastic']

# Stemming our data
ps = PorterStemmer()
features = features.apply(lambda x: x.split())
features = features.apply(lambda x : ' '.join([ps.stem(word) for word in x]))

# vectorizing the data with maximum of 5000 features
from sklearn.feature_extraction.text import TfidfVectorizer
tv = TfidfVectorizer(max_features = 5000)
features = list(features)
features = tv.fit_transform(features).toarray()


# getting training and testing data
features_train, features_test, labels_train, labels_test = train_test_split(features, labels, test_size = .05, random_state = 0)


# model 1:-
# Using linear support vector classifier
lsvc = LinearSVC()
# training the model
lsvc.fit(features_train, labels_train)
# getting the score of train and test data
print(lsvc.score(features_train, labels_train)) # 90.93
print(lsvc.score(features_test, labels_test))   # 83.75


# model 2:-
# Using Gaussuan Naive Bayes
gnb = GaussianNB()
gnb.fit(features_train, labels_train)
print(gnb.score(features_train, labels_train))  # 78.86
print(gnb.score(features_test, labels_test))    # 73.80


# model 3:-
# Logistic Regression
lr = LogisticRegression()
lr.fit(features_train, labels_train)
print(lr.score(features_train, labels_train))   # 88.16
print(lr.score(features_test, labels_test))     # 83.08


# model 4:-
# Random Forest Classifier
rfc = RandomForestClassifier(n_estimators = 10, random_state = 0)
rfc.fit(features_train, labels_train)
print(rfc.score(features_train, labels_train))  # 98.82
print(rfc.score(features_test, labels_test))    # 79.71


def predict(val):
    d = {0 : [val]}
    predict_data = pd.DataFrame(d)
    predict_string = predict_data[0].apply(lambda s : re.sub("[^a-zA-Z]",' ',s))
    predict_string = predict_string.apply(lambda x: x.split())
    predict_string = predict_string.apply(lambda x: ' '.join([ps.stem(word) for word in x]))
    predict_string = tv.transform(predict_string).toarray()
    return "Sarcastic" if lsvc.predict(predict_string)[0] == 1 else "Not Sarcastic"

print(predict("woman missing since she got lost"))



