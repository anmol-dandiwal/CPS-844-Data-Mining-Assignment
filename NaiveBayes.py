#!/usr/bin/python
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
import pandas as pd 

def NaiveBayes(dataset):
    atrributes = dataset.iloc[:, :-1]
    classes = dataset.iloc[:, 8]
    att_train, att_test, class_train, class_test = train_test_split(atrributes, classes, test_size=0.25)

    classifier = GaussianNB()

    model = classifier.fit(att_train, class_train)
    predictions = classifier.predict(att_test)

    print(accuracy_score(class_test, predictions))    
