#!/usr/bin/python
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, confusion_matrix, classification_report

def LogRegression(att_train, att_test, class_train, class_test):
    classifier = LogisticRegression()
    classifier.fit(att_train, class_train)

    predictions = classifier.predict(att_test)

    print('Logistic Regression: \nConfusion Matrix:\n',confusion_matrix(class_test, predictions),'\n')
    print(classification_report(class_test, predictions))
    print('F1 Score: ',f1_score(class_test, predictions),'\n\n')