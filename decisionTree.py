import numpy
import numpy as np
import csv
from collections import Counter
import math



def gini(data):  # gini of a choice
    game_results = data[:, -1]
    total = len(game_results)
    c = Counter(game_results)
    return 1 - sum((i[1] / total) ** 2 for i in c.items())


def gini_gain(data, index,possibleChoices):  # quality of split based on index
    if len(data) == 0:
        return -1
    choices = possibleChoices[index]

    gain = 0
    for choice in choices:
        choiceData = data[data[:, index] == choice]
        gain += (len(choiceData) / len(data) * gini(choiceData))
    return gain


def create_decision_tree(treeData, features,possibleChoices):
    outputSet = list(set(treeData[:, -1]))
    if len(treeData) == 0:
        return "NO DATA"
    if len(outputSet) == 1:  ##laplacian smoothing if nothing for this output
        return outputSet[0]
    if len(features) == 1:
        return Counter(list(treeData[:, -1])).most_common()[0][0]
    features = features.copy()
    splitIndex = sorted(features, key=lambda i: gini_gain(treeData, i,possibleChoices), reverse=False)[0]
    features.remove(splitIndex)
    d = {splitIndex: dict()}
    for choice in possibleChoices[splitIndex]:
        newdata = treeData[treeData[:, splitIndex] == choice]
        decision = create_decision_tree(newdata, features,possibleChoices)
        d[splitIndex][choice] = decision
    return d


def print_decision_tree(tree, depthLimit, depth=0):
    if not isinstance(tree, dict):
        print("\t" * (2 * depth) + tree)
        return
    if (depth == depthLimit):
        print("\t" * (2 * depth) + "{Tree Object}")
        return

    for key, value in tree.items():
        print("\t" * (2 * depth) + "CHECK INDEX " + str(key) + ":")
        for k, v in value.items():
            print("\t" * (2 * depth + 1) + "IF " + str(k) + " THEN:")
            print_decision_tree(v, depthLimit, depth=depth + 1)


def evaluate_decision_tree(data_instance, tree):
    if not isinstance(tree, dict):
        return tree

    for key, value in tree.items():
        instance_value = data_instance[key]
        return evaluate_decision_tree(data_instance, value[instance_value])

if __name__ == "__main__":
    # print(1/5)
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

    # print(data.shape)
    possibleChoices = []
    for col in range(trainingData.shape[1]):
        children = list(set(data[:, col]))
        possibleChoices.append(children)
        # print(possibleChoices[col])


    final_tree = create_decision_tree(trainingData, list(range(np.shape(trainingData)[1] - 1)),possibleChoices)
    print_decision_tree(final_tree, 99)


    c1 = (Counter([evaluate_decision_tree(i, final_tree)==i[-1] for i in testingData]))
    print(c1)
    print([i[1]  for i in c1.items()])

