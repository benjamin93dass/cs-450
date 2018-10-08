import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics
import numpy as np


# Clean up the data
def cleanData(dataframe, targetCol):

    # Remove all 'NAN' values
    dataframe = dataframe.fillna(0)

    # Replace all '?' to '0' as '?' is treated as 'NAN'
    dataframe.replace(to_replace='?', value=0, inplace=True)

    # Mapping each element with an int and replacing it
    for col in dataframe.keys():

        # get a list of unique elements from the dataframe's column
        uElem = dataframe[col].unique()

        # for each unique element in the list
        for i in range(len(uElem)):

            # Replacing the unique element with an index
            dataframe[col].replace(to_replace=uElem[i], value=(i + 1), inplace=True)

    # Retrieving target columns
    target = dataframe[targetCol]

    # Delete the column from the dataframe
    del dataframe[targetCol]

    return dataframe, target


# Loading data from the file, parsing it after
def loadData(filename, delim):

    # Reading the file
    with open(filename, "r") as file:

        # While reading from file, split the line into words and append them into a data list
        data = [[y for y in x.strip().split('\t')[0].strip().split(delim) if y != ""] for x in file.readlines()]

    # Returning converted dataframe from a data list.
    return pd.DataFrame(data)


# Start predicting the data given
def runKnnClassification(kMin, kMax, dataset, target, targetCol):

    # Generates floats with an increment of step
    def floatRange(start, stop, step):
        i = start
        while i < stop:
            yield round(i, 2)
            i += step

    # Retreiving test data split value
    for split in floatRange(0.1, 0.6, 0.05):

        print("Train : Test = {:.2f} : {:.2f}".format(1 - split, split))

        # Splitting the dataset + target into training and test sets
        trainSet, testSet, trainTarget, testTarget = train_test_split(dataset, target, test_size=split, random_state=42)

        # Testing each number of k
        for k in range(kMin, kMax):

            print("   K = {:d}", k)

            # Creating a KNN Classifier
            knn = KNeighborsClassifier(n_neighbors=k)

            # Fitting train and target to KNN Model
            knn.fit(trainSet, trainTarget)

            # Predicting train + calculating accuracy of test set
            predictions = knn.predict(trainSet)
            accuracy = metrics.accuracy_score(trainTarget, predictions) * 100
            print("\t  Train : {:.2f}%".format(accuracy))

            # Predicting + calculating accuracy of test set
            predictions = knn.predict(testSet)
            accuracy = metrics.accuracy_score(testTarget, predictions) * 100
            print("\t  Test : {:.2f}%".format(accuracy))


# Initializing kMin and kMax
kMin, kMax = 1, 10

# 1) Getting the file + printing it out what file I am in,
# 2) Start loading the data from the following file,
# 3) Setting targets,
# 4) Make sure the data is clean,
# 5) Run the KNN Classification with different variations,
# 6) These are repeated 3 times with car, autism and auto-mpg.

filename = 'car.data'
print('Knn Classification for {:s} : '.format(filename))
dataset = loadData(filename, ',')
targetCol = 6
dataset, target = cleanData(dataset, targetCol)
runKnnClassification(kMin, kMax, dataset, target, 6)

filename = 'autism.arff'
print('Knn Classification for {:s} : '.format(filename))
dataset = loadData(filename, ',')
targetCol = 20
dataset, target = cleanData(dataset, targetCol)
runKnnClassification(kMin, kMax, dataset, target, 20)

filename = 'auto-mpg.data'
print('Knn Classification for {:s} : '.format(filename))
dataset = loadData(filename, ' ')
targetCol = 0
dataset, target = cleanData(dataset, targetCol)
runKnnClassification(kMin, kMax, dataset, target, 0)
