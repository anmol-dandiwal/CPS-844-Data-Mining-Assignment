#!/usr/bin/python
from sklearn.model_selection import train_test_split
from sklearn import tree
from sklearn.metrics import accuracy_score

def DecisionTree(dataset):
    atrributes = dataset.iloc[:, :-1]
    classes = dataset.iloc[:, 8]
    att_train, att_test, class_train, class_test = train_test_split(atrributes, classes, test_size=0.25)

    classifier = tree.DecisionTreeClassifier(max_depth=3)
    classifier.fit(att_train, class_train)
    
    print('Decisiion Tree:\n',tree.export_text(classifier, feature_names=dataset.columns.tolist()[:-1]),'\n')
    
    predictions = classifier.predict(att_test)

    print('Accuracy: ',accuracy_score(class_test, predictions),'\n')    