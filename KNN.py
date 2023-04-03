#!/usr/bin/python
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import f1_score, confusion_matrix, classification_report
import pandas as pd 

def kNearestNeighbours(k):
    url = 'pulsars.csv'
    names = ['att1', 'att2', 'att3', 'att4', 'att5', 'att6', 'att7', 'att8', 'Class']
    dataset = pd.read_csv(url, names=names)

    atrributes = dataset.iloc[:, :-1]
    classes = dataset.iloc[:, 8]
    att_train, att_test, class_train, class_test = train_test_split(atrributes, classes, test_size=0.20)
    scaler = StandardScaler()
    
    scaler.fit(att_train)
    att_train = scaler.transform(att_train)
    att_test = scaler.transform(att_test)

    classifier = KNeighborsClassifier(k)
    classifier.fit(att_train, class_train)

    predictions = classifier.predict(att_test)

    print('Confusion Matrix:\n', confusion_matrix(class_test, predictions), '\n')
    print(classification_report(class_test, predictions))
    print('F1 Score: ', f1_score(class_test, predictions))