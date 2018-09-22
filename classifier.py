import numpy as np
import scipy
from sklearn import datasets
from sklearn.cross_validation import train_test_split
from sklearn.naive_bayes import GaussianNB

iris = datasets.load_iris()

# Show the data (attributes of each instance)
# print("The whole iris dataset:")
# print(iris.data)

# Show the target values of each instance
# y = iris.target
# print("\nTarget values:")
# print(iris.target)

# Show the actual target names that correspond to each number
# print("\nTarget names:")
# print(iris.target_names)

# Preparing training/test sets
X_train, X_test, y_train, y_test = train_test_split(iris.data, iris.target, test_size=0.33, random_state=1)

# using gaussianNB
classifier = GaussianNB()
model = classifier.fit(X_train, y_train)

# start making predictions
y_predicted = model.predict(X_test)

# comparing results
cor = 0
err = 0
for i in range(0, len(y_predicted)):
    if y_predicted[i] == y_test[i]:
        cor += 1
    else:
        err += 1

percentError = 100 * (err / (err + cor))

print("Sum correct: {}".format(cor))
print("Sum error: {}".format(err))
print("Percent error: {}%".format(percentError))


# creating my new algorithm
class HardCodedClassifier:
    def fit(self, X_train, y_train):
        model = HardCodedModel()
        return model


class HardCodedModel:
    def predict(self, data_test):
        length = len(data_test)
        predictions = np.zeros((length,))
        return predictions


classifier = HardCodedClassifier()
model = classifier.fit(X_train, y_train)
y_predict = model.predict(X_train)

correct = 0
error = 0
for k in range(0, 50):
    if y_predict[k] == y_test[k]:
        correct += 1
    else:
        error += 1

percentError = 100 * (error / (error + correct))

print("\nSum correct: {}".format(correct))
print("Sum error: {}".format(error))
print("Percent error: {}%".format(percentError))

print("\nY_predicted: ", y_predicted)

# Above and Beyond
# accomplishing n-folds
# learnt this from http://scikit-learn.org/stable/modules/cross_validation.html
from sklearn.model_selection import cross_val_score
from sklearn import svm

clf = svm.SVC(kernel='linear', C=1)
scores = cross_val_score(clf, iris.data, iris.target, cv=5)
print("\nScores: {}".format(scores))
print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))


