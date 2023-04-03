#!/usr/bin/python
from KNN import kNearestNeighbours

def main():
    # After checking all values for k from 1-10, 5 was most accurate
    kNearestNeighbours(5)

if __name__ == '__main__':
    main()