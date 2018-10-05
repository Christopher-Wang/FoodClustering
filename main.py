import numpy as np
import pandas as pd
import kNN
import math
import Unpackage_Data as data

def euclideanDistance(instance1, instance2, columns):
	distance = 0
	for column in columns:
		distance += pow((instance1[column] - instance2[column]), 2)
	return math.sqrt(distance)

def getAccuracy(testSet, predictions):
	correct = 0
	for x in range(len(testSet)):
		if testSet[x][-1] == predictions[x]:
			correct += 1
	return (correct/float(len(testSet))) * 100.0


def main():
	dataframe = data.getData()
	dataframe = dataframe.get_values()
	mask = np.random.rand(len(dataframe)) < 0.8
	trainingData, testingData = dataframe[mask], dataframe[~mask]
	model = kNN.kNN(8, euclideanDistance, trainingData)
	columns = [x for x in range(3, 11)]

	predictions = model.getPredictions(testingData, columns)
	print(getAccuracy(testingData, predictions))


main()
