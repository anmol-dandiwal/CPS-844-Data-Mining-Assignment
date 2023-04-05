#!/usr/bin/python
from sklearn import tree
from sklearn.metrics import f1_score, confusion_matrix, classification_report

def DecisionTree(att_train, att_test, class_train, class_test, dataset):
    classifier = tree.DecisionTreeClassifier(max_depth=3)
    classifier.fit(att_train, class_train)
    
    print('Decisiion Tree:\n',tree.export_text(classifier, feature_names=dataset.columns.tolist()[:-1]),'\n')
    
    predictions = classifier.predict(att_test)

    print('Confusion Matrix:\n',confusion_matrix(class_test, predictions),'\n')
    print(classification_report(class_test, predictions))
    print('F1 Score: ',f1_score(class_test, predictions),'\n\n')  