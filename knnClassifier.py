import numpy as np
from sklearn import datasets
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import cross_val_score, train_test_split

# Loading dataset
from sklearn.neighbors import KNeighborsClassifier

iris = datasets.load_iris()

# Preparing training/test sets
data_train, data_test, target_train, target_test = train_test_split(iris.data, iris.target, test_size=0.33,
                                                                    random_state=1)

# using gaussianNB
classifier = GaussianNB()
model = classifier.fit(data_train, target_train)

# start making predictions
targets_predicted = model.predict(data_test)

data_list = iris.data.tolist()
data_test_list = data_test.tolist()

# comparing results
cor = 0
err = 0
for i in range(len(data_test)):
    if iris.target[data_list.index(data_test_list[i])] == targets_predicted[i]:
        cor += 1

print('Accuracy: {}/{} => {}%'.format(cor, len(data_test), (cor / len(data_test) * 100)))

print('\n***************************')
print('       Self made KNN')
print('***************************')


# knnClassifier
class HardCodedModel:
    def __get_distances(self, test_ele):
        distances_array = []

        for x in self.data_trained:
            # learnt this cool thing from https://plot.ly/numpy/norm/
            # this computes matrix norm, vector spaces
            dist = np.linalg.norm(test_ele - x)

            # start appending distances to the array
            distances_array = np.append(distances_array, dist)

        # Adding targets to distances
        distances_array = np.array([distances_array, self.target_trained])
        distances_array = np.swapaxes(distances_array, 0, 1)

        # Sorting array by distance
        distances_array = distances_array[distances_array[:, 0].argsort()]

        return distances_array

    def __init__(self, data, target):
        self.data_trained = np.array(data)
        self.target_trained = np.array(target)

    def predict(self, test, k):
        target = []

        for elements in test:
            # Grabbing elements
            distances = self.__get_distances(elements)

            # Getting the first k distances
            possible_targets = distances[:k, 1]

            # Getting the frequencies of types
            values, counts = np.unique(possible_targets, return_counts=True)

            # Creating an array for targets and its frequencies
            value_count = np.array([values, counts])
            value_count = np.swapaxes(value_count, 0, 1)

            # Start sorting by frequency
            value_count = value_count[value_count[:, 1].argsort()]

            # Adding the most frequent target
            target = np.append(target, value_count[::-1, 0][0])

        target = target.astype(int)
        return target


class knnClassifier:
    def fit(self, data, target):
        return HardCodedModel(data, target)


# trying out different number of neighbours
k = 9
classifier = knnClassifier()
model = classifier.fit(data_train, target_train)
targets_predicted = model.predict(data_test, k)

print('Where k = {},'.format(k))

corr = 0
for i in range(len(data_test)):
    if iris.target[data_list.index(data_test_list[i])] == targets_predicted[i]:
        corr += 1

print('Accuracy: {}/{} => {} %'.format(corr, len(data_test), (corr / len(data_test)) * 100))

k = 4
classifier = knnClassifier()
model = classifier.fit(data_train, target_train)
targets_predicted = model.predict(data_test, k)

print('\nWhere k = {},'.format(k))

corr = 0
for i in range(len(data_test)):
    if iris.target[data_list.index(data_test_list[i])] == targets_predicted[i]:
        corr += 1

print('Accuracy: {}/{} => {} %'.format(corr, len(data_test), (corr / len(data_test)) * 100))

print('\n***************************')
print('       SKLearn KNN')
print('***************************')

print('nWhere k = 3,')

# Using SKLearns knn algorithm
classifier = KNeighborsClassifier(n_neighbors=3)
model = classifier.fit(data_train, target_train)
targets_predicted = model.predict(data_test)

corr = 0

for i in range(len(data_test)):
    if iris.target[data_list.index(data_test_list[i])] == targets_predicted[i]:
        corr += 1

print('Accuracy: {}/{} => {} %'.format(corr, len(data_test), (corr / len(data_test)) * 100))