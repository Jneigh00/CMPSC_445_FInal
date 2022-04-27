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

# Split training data into X and Y arrays
X_train_string = np.delete(trainingData, -1, 1)
Y_train_string = trainingData[:, -1]

# Arrays for the training data that is numerical
X_train = X_train_string.astype(float)
Y_train = numpy.empty(len(Y_train_string))

# Changing the values of the output array, 1 if won, 0 if no win
for i in range(len(Y_train_string)):
    case = Y_train_string[i]
    if case == 'won':
        Y_train[i] = 1
    if case == 'nowin':
        Y_train[i] = 0

# Converting the training data to a float array
trainingDataNoString = trainingData
for i in range(len(trainingData)):
    for j in range(len(trainingData[i])):
        element = trainingData[i][j]
        if element == 0:
            trainingDataNoString[i][j] = 0
        if element == 1:
            trainingDataNoString[i][j] = 1
        if element == 'won':
            trainingDataNoString[i][j] = 1
        if element == 'nowin':
            trainingDataNoString[i][j] = 0
trainingDataNoString = trainingDataNoString.astype(float)


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
ann.fit(X_train, Y_train, batch_size=32, epochs=25)

# Prediction testing
# print(ann.predict(trainingDataNoString[0]))

accuracy = numpy.array([0.5818, 0.6482, 0.7371, 0.8210, 0.8831, 0.9098, 0.9336, 0.9445, 0.9528, 0.9558, 0.9633, 0.9641,
                        0.9695, 0.9725, 0.9754, 0.9783, 0.9808, 0.9812, 0.9862, 0.9841, 0.9887, 0.9879, 0.9891, 0.9908,
                        0.9896])
epochs = numpy.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25])

plt.plot(epochs,accuracy)
plt.title('ANN Accuracy For One Sample Run')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.show()


