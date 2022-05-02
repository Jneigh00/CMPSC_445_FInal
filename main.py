import sys

import numpy
import numpy as np
import csv
import sys
import pandas as pd
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense, Activation
import matplotlib.pyplot as plt

# numpy.set_printoptions(threshold=sys.maxsize)
# print(tf.__version__)
from decisionTree import create_decision_tree, print_decision_tree, evaluate_decision_tree
from collections import Counter
import functools
from naiveBayes import naiveBayes

np.set_printoptions(linewidth=320)

# Reads in the csv file and stores it into a list
with open("kr-vs-kp_csv.csv") as file_name:
    file_read = csv.reader(file_name)
    array = list(file_read)

# converts the list into a numpy array
data = numpy.array(array)
columnLabels = data[0]
data = data[1:]  # separate the column labels from the rest of the data
np.random.seed(2)  # use a predetermined seed so the shuffle order is always the same
np.random.shuffle(data)  # shuffle the order of the data, for a more fair sample distribution

# loops through each row to change the values
for x in range(len(data)):
    for y in range(len(data[x])):
        element = data[x][y]
        if element == 'f' or element == 'l' or (
                y == 35 and element == 'n'):  # if the current element is a f or a l then it changes it to a zero
            data[x][y] = 0
        if element == 't' or element == 'g':  # if the current element is a t or a g then it changes it to a one
            data[x][y] = 1
        if element == 'n':
            data[x][y] = 0
        if element == 'w':
            data[x][y] = 1
        if element == 'b':
            data[x][y] = -1

splitIndex = int(0.75 * len(data))  # gets the index to split the data 75/25

trainingData = data[1:splitIndex]  # stores 75 percent of the data into training
testingData = data[splitIndex:]  # stores remaining 25 percent of data into testing

def print_accuracies(testData, decision_func,alg_name):
    accuracy = []
    results = ["False Positive","False Negative",  "True Negative", "True Positive"]
    for instance in testData:
        decision = decision_func(instance)
        accuracy.append(results[2*int(decision == instance[-1])+int(decision == "won")])

    print("Accuracy data for",alg_name+":")
    c = Counter({x:0 for x in results})
    c.update(accuracy)
    print("Accuracy of Algorithm:",(c["True Negative"]+c["True Positive"])/len(testData))
    for k,v in sorted(c.items()):
        print(k,v)
    print()


######################## ANN CODE ################################

# Split training and testing data into X and Y arrays
X_train_string = np.delete(trainingData, -1, 1)
Y_train_string = trainingData[:, -1]

X_test_string = np.delete(testingData, -1, 1)
Y_test_string = testingData[:, -1]

# Arrays for the training and testing data that is numerical
X_train = X_train_string.astype(float)
Y_train = numpy.empty(len(Y_train_string))

X_test = X_test_string.astype(float)
Y_test = numpy.empty(len(Y_test_string))

# Changing the values of the output array, 1 if won, 0 if no win
for i in range(len(Y_train_string)):
    case = Y_train_string[i]
    if case == 'won':
        Y_train[i] = 1
    if case == 'nowin':
        Y_train[i] = 0

# Initializing ANN
ann = tf.keras.models.Sequential()

# Adding first hidden layer
ann.add(tf.keras.layers.Dense(units=6, activation="relu"))

# Adding second hidden layer
ann.add(tf.keras.layers.Dense(units=6, activation="relu"))

# Adding output layer
ann.add(tf.keras.layers.Dense(units=1, activation="sigmoid"))

# Compiling ANN
ann.compile(optimizer="adam", loss="binary_crossentropy", metrics=['accuracy'])

# Fitting ANN
ann.fit(X_train, Y_train, batch_size=32, epochs=75,verbose=0)

# Prediction testing
######print(ann.predict(X_test).ravel() > 0.5)
decisions = ann.predict(X_test)

accuracy_data = np.append(decisions, Y_test_string.reshape(-1,1), 1)

print_accuracies(accuracy_data, lambda x: "won" if float(x[-2])>0.5 else "nowin", "ANN")

# accuracy = numpy.array([0.5818, 0.6482, 0.7371, 0.8210, 0.8831, 0.9098, 0.9336, 0.9445, 0.9528, 0.9558, 0.9633, 0.9641,
#                         0.9695, 0.9725, 0.9754, 0.9783, 0.9808, 0.9812, 0.9862, 0.9841, 0.9887, 0.9879, 0.9891, 0.9908,
#                         0.9896])
# epochs = numpy.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25])
#
# plt.plot(epochs,accuracy)
# plt.title('ANN Accuracy For One Sample Run')
# plt.xlabel('Epoch')
# plt.ylabel('Accuracy')
# plt.show()

##################################################################



######################## DECISON TREE CODE ################################

possibleChoices = [list(set(data[:, col])) for col in range(trainingData.shape[1])]
final_tree = create_decision_tree(trainingData, list(range(np.shape(trainingData)[1] - 1)), possibleChoices)
# print_decision_tree(final_tree, 99)
print_accuracies(testingData, lambda instance: evaluate_decision_tree(instance, final_tree), "DECISION TREE")

###########################################################################

######################## NAIVE BAYES CODE  ################################
nb = naiveBayes(trainingData)
print_accuracies(testingData, naiveBayes(trainingData).decide, "NAIVE BAYES")

###########################################################################
