import functools

import numpy as np
import numpy
import csv
from collections import Counter

class naiveBayes:
    data = None
    def __init__(self, data):
        self.data = data

    @functools.lru_cache(maxsize=None)
    def prob_x_c(self, x, c, index):
        data = self.data
        c_data = data[data[:, -1] == c]
        px = Counter(c_data[:, index])
        return px[x] / len(c_data)

    def decide(self, x):
        data = self.data
        p_values = []
        for c in ["won", "nowin"]:
            p = Counter(data[:, -1])[c] / len(data)
            for index in range(len(data[0]) - 1):
                p *= self.prob_x_c(x[index], c, index)
            p_values.append(p)
        for i in range(len(p_values)):
            if p_values[i] == max(p_values):
                return ["won", "nowin"][i]


if __name__ == "__main__":
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

    splitIndex = int(0.75 * len(data))  # gets the index to split the data 75/25

    trainingData = data[1:splitIndex]  # stores 75 percent of the data into training
    testingData = data[splitIndex:]  # stores remaining 25 percent of data into testing

    nb = naiveBayes(trainingData)
    accuracy = []
    for instance in testingData:
        decision = nb.decide(instance)
        if (decision == instance[-1]):
            if (decision == "won"):
                accuracy.append("True Positive")
            else:
                accuracy.append("True Negative")
        else:
            if (decision == "nowin"):
                accuracy.append("False Positive")
            else:
                accuracy.append("False Negative")
    print(Counter(accuracy))
