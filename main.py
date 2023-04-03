#!/usr/bin/python
from KNN import KNearestNeighbours
from NaiveBayes import NaiveBayes
import pandas as pd

def main():
    url = 'pulsars.csv'
    names = ['att1', 'att2', 'att3', 'att4', 'att5', 'att6', 'att7', 'att8', 'Class']
    dataset = pd.read_csv(url, names=names)

    # After checking all values for k from 1-10, 5 was most accurate
    KNearestNeighbours(dataset, 5)

    NaiveBayes(dataset)    

if __name__ == '__main__':
    main()