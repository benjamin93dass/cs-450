from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import f1_score
from math import sqrt
import operator
from collections import Counter
from pandas import *
from types import *
import numpy
import random
import math
import matplotlib.pyplot as plt

def determine_accuracy(test_target, targets_predicted):
    correct = 0
    total = 0
    for x in range(len(test_target)):
        if test_target.iloc[x] == targets_predicted[x]:
            correct = correct + 1
        total = total + 1
    percent = (correct * 100) / total

    print("Total correct: (", correct, "/", total, "): ", float("{0:.2f}".format(percent)), "%")

def ExampleNeuralNet(data):
    training, test = train_test_split(data, test_size=0.30)
    training_data = training.loc[:, test.columns != "target"]
    training_target = training["target"]
    test_data = test.loc[:, test.columns != "target"]
    test_target = test["target"]

    clf = MLPClassifier(solver='lbfgs', alpha=1e-5,  hidden_layer_sizes = (2), random_state = 1)
    clf.fit(training_data, training_target)
    targets_predicted = clf.predict(test_data)
    determine_accuracy(test_target, targets_predicted)

class Neural_Node:
    def __init__(self,num_attributes):
        self.weights = {}
        self.bias = -1
        self.actual = 0
        self.error = 0
        for i in range(num_attributes):
            self.weights[i] = random.uniform(-1.0, 1.0)
        self.weights["bias"] =random.uniform(-1.0, 1.0)

    def output(self,instant):
        output = 0
        i = 0
        for attribute in instant:
            output = output + (self.weights[i] * attribute)
            i = i + 1
        output = output + (self.weights["bias"] * self.bias)
        #https://stackoverflow.com/questions/44466751/sigmoid-function-in-python
        output = 1 / (1 + math.e ** -output)
        self.actual = output
        return output

    def output_error_calc(self,expected):
        self.error = self.actual * (1 - self.actual) * (self.actual - expected)
        error_array = []
        for i in range(len(self.weights) - 1):
            error_array.append(self.weights[i] * self.error)
        return error_array

    def error_calc(self,weighted_error):
        self.error = self.actual * (1 - self.actual) * weighted_error
        error_array = []
        for i in range(len(self.weights) - 1):
            error_array.append(self.weights[i] * self.error)
        return error_array

    def update_weights(self,instant,learning_rate):
        i = 0
        for attribute in instant:
            self.weights[i] = self.weights[i] - (attribute * self.error * learning_rate)
            i = i + 1
        self.weights["bias"] = self.weights["bias"] - (self.bias * self.error * learning_rate)
        return self.actual

class Neural_Layer:
    def __init__(self,num_nodes,num_weights, isOutput = False):
        self.num_nodes = int(num_nodes)
        self.num_weights = num_weights
        self.nodes = []
        self.isOutput = isOutput
        self.errors = []
        for i in range(self.num_nodes):
            self.nodes.append(Neural_Node(self.num_weights))
            self.errors.append(0)

    def output(self,instant):
        layer_outputs = []
        for node in self.nodes:
            layer_outputs.append(node.output(instant))
        return layer_outputs

    def output_error_calc(self,expected):
        node_num = 0
        for node in self.nodes:
            if node_num == expected:
                self.errors[node_num] = node.output_error_calc(1)
            else:
                self.errors[node_num] = node.output_error_calc(0)
            node_num = node_num + 1
        return self.errors

    def error_calc(self,errors):
        layer_nodes_error = []
        for i in range(len(self.nodes)):
            layer_nodes_error.append(0)
        for error in (errors):
            for i in range(len(error)):
                layer_nodes_error[i] = error[i] + layer_nodes_error[i]
        for i in range(len(self.nodes)):
            self.errors[i] = self.nodes[i].error_calc(layer_nodes_error[i])
        return self.errors

    def update_weights(self,instant,learning_rate):
        layer_outputs = []
        for node in self.nodes:
            layer_outputs.append(node.update_weights(instant,learning_rate))
        return layer_outputs

class Neural_Net:
    def __init__(self,num_nodes,num_attributes,unique_targets):
        self.layers = []
        self.num_weights = num_attributes
        self.unique_targets = unique_targets
        for i in num_nodes:
            self.layers.append(Neural_Layer(i,self.num_weights))
            self.num_weights = int(i)
        self.layers.append(Neural_Layer(self.unique_targets.size,self.num_weights,True))

    def error_calc(self, expected):
        for layer in reversed(self.layers):
            if layer.isOutput == True:
                errors = layer.output_error_calc(expected)
            else:
                errors = layer.error_calc(errors)

    def predict(self,instant):
        for layer in self.layers:
            instant = layer.output(instant)

        return self.unique_targets[instant.index(max(instant))]

    def learn(self,instance, learning_rate):
        for layer in self.layers:
            instance = layer.update_weights(instance,learning_rate)

class MyNeuralNetModel:
    def __init__(self,neural_net):
        self.neural_net = neural_net
    def predict(self,data):
        predicted_targets = []
        for index, row in data.iterrows():
            predicted_targets.append(self.neural_net.predict(row))
        return predicted_targets

class MyNeuralNetClassifier:
    def __init__(self,num_nodes,num_attributes,unique_targets):
        self.neural_net = Neural_Net(num_nodes,num_attributes,unique_targets)

    def fit(self,training_data,training_targets,learning_rate,num_epochs,stopping_percentage):
        epoch = []
        for i in range (num_epochs):
            correct = 0
            for index, row in training_data.iterrows():
                actual = self.neural_net.predict(row)
                if training_targets.loc[index] != actual:
                    self.neural_net.error_calc(training_targets.loc[index])
                    self.neural_net.learn(row,learning_rate)
                else:
                    correct = correct + 1
            percentage = correct / training_targets.size
            if (percentage > stopping_percentage):
                break
            epoch.append(correct)

        plt.plot(epoch)
        plt.ylabel('Accuracy')
        plt.show()
        model = MyNeuralNetModel(self.neural_net)
        return model

def MyNeuralNet(data):
    unique_targets = unique(data["target"])
    training, test = train_test_split(data, test_size=0.30)
    training_data = training.loc[:, test.columns != "target"]
    training_targets = training["target"]
    test_data = test.loc[:, test.columns != "target"]
    test_targets = test["target"]

    epoch = input("Enter a number of training epochs: ")
    epoch = int(epoch)
    learning_rate = input("Enter a learning rate: ")
    learning_rate = float(learning_rate)
    stopping_percentage = input("Enter a stopping percentage. Enter 0 if desired:")
    stopping_percentage = float(stopping_percentage)
    num_layers = input("Enter number of layers: ")
    num_layers = int(num_layers)
    num_nodes = []
    for i in range(num_layers):
        print("Layer ",i,": ")
        num_nodes.append(input("Enter number of nodes for this layer: "))

    classifier = MyNeuralNetClassifier(num_nodes, len(data.columns) - 1,unique_targets)
    model = classifier.fit(training_data,training_targets,learning_rate,epoch,stopping_percentage)
    predicted_targets = model.predict(test_data)
    determine_accuracy(test_targets, predicted_targets)

def Load_Iris():
    iris_bunch = datasets.load_iris()
    iris = datasets.load_iris()
    numpy.savetxt("iris.csv", iris.data, delimiter=",")
    data = read_csv("iris.csv", header=0)
    data.columns = [0,1,2,3]
    data["target"] = pandas.Series(iris.target)

    return data

def Load_Indian():
    data = read_csv("Indian data.csv", header=0)
    data.columns = ["Pregnant", "Glucose", "Blood_Pressure", "Skin_Thick", "Insulin", "BMI", "Pedigree", "Age", "target"]

    data["Glucose"] = data["Glucose"].replace(0, 120)
    data["Blood_Pressure"] = data["Blood_Pressure"].replace(0, 69.1)
    data["Skin_Thick"] = data["Skin_Thick"].replace(0, 20.5)
    data["Insulin"] = data["Insulin"].replace(0, 79.8)
    data["BMI"] = data["BMI"].replace(0, 32.0)
    data["Age"] = data["Age"].replace(0, 33.2)

    data["Pregnant"] = data["Pregnant"] / 3.4
    data["Glucose"] = data["Glucose"] / 32.0
    data["Blood_Pressure"] = data["Blood_Pressure"] / 19.4
    data["Skin_Thick"] = data["Skin_Thick"] / 16.0
    data["Insulin"] = data["Insulin"] / 115.2
    data["BMI"] = data["BMI"] / 7.9
    data["Pedigree"] = data["Pedigree"] / 0.3
    data["Age"] = data["Age"] / 11.8

    return data

def Load_Car():
    data = read_csv("car data.csv", header=0)
    data.columns = ["Buying","Maint","Doors","Persons","Lug_boot","Safety","target"]
    buying = ["vhigh", "high", "med", "low"]
    maInt = ["vhigh", "high", "med", "low"]
    doors = ["2", "3", "4", "5more"]
    persons = ["2", "4", "more"]
    lug_boot = ["small", "med", "big"]
    safety = ["low", "med", "high"]
    class_value = ["unacc", "acc", "good", "vgood"]

    for x in range(len(buying)):
        data["Buying"] = data["Buying"].replace(buying[x], x)

    for x in range(len(maInt)):
        data["Maint"] = data["Maint"].replace(maInt[x], x)

    for x in range(len(doors)):
        data["Doors"] = data["Doors"].replace(doors[x], x)

    for x in range(len(persons)):
        data["Persons"] = data["Persons"].replace(persons[x], x)

    for x in range(len(lug_boot)):
        data["Lug_boot"] = data["Lug_boot"].replace(lug_boot[x], x)

    for x in range(len(safety)):
        data["Safety"] = data["Safety"].replace(safety[x], x)

    for x in range(len(class_value)):
        data["target"] = data["target"].replace(class_value[x], x)

    return data

iris_data = Load_Iris()
#indian_data = Load_Indian()
#car_data = Load_Car()
MyNeuralNet(iris_data)
#MyNeuralNet(indian_data)
#MyNeuralNet(car_data)
print("Off the shelf inplementation: ")
ExampleNeuralNet(iris_data)
#ExampleNeuralNet(indian_data)
#ExampleNeuralNet(car_data)