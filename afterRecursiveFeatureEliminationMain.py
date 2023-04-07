#!/usr/bin/python
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from Classifiers.KNN import KNearestNeighbours
from Classifiers.NaiveBayes import NaiveBayes
from Classifiers.LogRegression import LogRegression
from Classifiers.DecisionTree import DecisionTree
from Classifiers.SVM import SVM
import pandas as pd

#IP = Integrated Profile
def main():
    url = 'recursiveFeatureEliminationPulsar.csv'
    #Names for [2, 3, 4, 5]
    names = ['Excess kurtosis of IP', 'Skewness of IP', 'Mean of the DM-SNR', 'Standard deviation of DM-SNR','Class']
    dataset = pd.read_csv(url, names=names)

    atrributes = dataset.iloc[:, :-1]
    classes = dataset.iloc[:, 4]
    att_train, att_test, class_train, class_test = train_test_split(atrributes, classes, test_size=0.25)

    scaler = StandardScaler()
    scaler.fit(att_train)
    att_train = scaler.transform(att_train)
    att_test = scaler.transform(att_test)

    # After checking all values for k from 1-10, 5 was most accurate
    KNearestNeighbours(att_train, att_test, class_train, class_test, 5)
    NaiveBayes(att_train, att_test, class_train, class_test)
    LogRegression(att_train, att_test, class_train, class_test)
    DecisionTree(att_train, att_test, class_train, class_test, dataset)
    SVM(att_train, att_test, class_train, class_test)

if __name__ == '__main__':
    main()