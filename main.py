import numpy
import numpy as np
import csv


#Reads in the csv file and stores it into a list
with open("kr-vs-kp_csv.csv") as file_name:
    file_read = csv.reader(file_name)
    array = list(file_read)

#converts the list into a numpy array
data = numpy.array(array)

#loops through each row to change the values
for x in range(len(data)):
    for y in range(len(data[x])):
        element = data[x][y]
        if element == 'f' or element == 'l': #if the current element is a f or a l then it changes it to a zero
            data[x][y] = 0
        if(element == 't' or element == 'g'): #if the current element is a t or a g then it changes it to a one
            data[x][y] = 1


splitIndex = int(0.75*len(data)) #gets the index to split the data 75/25
trainingData = data[:splitIndex] #stores 75 percent of the data into training
testingData = data[splitIndex:]  #stores remaining 25 percent of data into testing

print(trainingData)
print("This is new line")
print(testingData)



