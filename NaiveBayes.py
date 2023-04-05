#!/usr/bin/python
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import f1_score, confusion_matrix, classification_report

def NaiveBayes(att_train, att_test, class_train, class_test):
    classifier = GaussianNB()
    classifier.fit(att_train, class_train)
    
    predictions = classifier.predict(att_test)

    print('Naive Bayes: \nConfusion Matrix:\n',confusion_matrix(class_test, predictions),'\n')
    print(classification_report(class_test, predictions))
    print('F1 Score: ',f1_score(class_test, predictions),'\n\n')
