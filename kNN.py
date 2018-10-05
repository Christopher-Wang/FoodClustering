# Example of kNN implemented from Scratch in Python
import csv
import random
import operator


class kNN(object):
	def __init__(self, k, distance, data):
		self.k = k
		self.distance = distance
		self.data = data

	def getNeighbors(self, testInstance, columns):
		distances = []
		for x in range(len(self.data)):
			dist =self.distance(testInstance, self.data[x], columns)
			distances.append((self.data[x], dist))
		distances.sort(key=operator.itemgetter(1))
		neighbors = []
		for x in range(self.k):
			neighbors.append(distances[x][0])
		return neighbors
	 
	def getResponse(self, neighbors):
		classVotes = {}
		for x in range(len(neighbors)):
			response = neighbors[x][-1]
			if response in classVotes:
				classVotes[response] += 1
			else:
				classVotes[response] = 1
		sortedVotes = max(classVotes.items(), key=operator.itemgetter(1))[0]
		return sortedVotes

	def getPrediction(self, testInstance, columns):
		neighbors = self.getNeighbors(testInstance, columns)
		response = self.getResponse(neighbors)
		return response

	def getPredictions(self, testSet, columns):
		predictions = []
		for testInstance in testSet:
			predictions.append(self.getPrediction(testInstance, columns))
		return predictions