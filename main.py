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


######################## DECISON TREE CODE ################################

# print(data.shape)
possibleChoices = [list(set(data[:, col])) for col in range(trainingData.shape[1])]

final_tree = create_decision_tree(trainingData, list(range(np.shape(trainingData)[1] - 1)),possibleChoices)
#print_decision_tree(final_tree, 99)
for instance in testingData:
    decision = evaluate_decision_tree(instance,final_tree)
    ##TODO: decision is the choice made by the decision tree for each row in testingData, either "won" or "nowin"

###########################################################################

