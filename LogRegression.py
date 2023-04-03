#!/usr/bin/python
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

def LogRegression(dataset):
    atrributes = dataset.iloc[:, :-1]
    classes = dataset.iloc[:, 8]
    att_train, att_test, class_train, class_test = train_test_split(atrributes, classes, test_size=0.25)

    scaler = StandardScaler()
    scaler.fit(att_train)
    att_train = scaler.transform(att_train)
    att_test = scaler.transform(att_test)

    classifier = LogisticRegression()
    classifier.fit(att_train, class_train)
    
    predictions = classifier.predict(att_test)

    print(accuracy_score(class_test, predictions))