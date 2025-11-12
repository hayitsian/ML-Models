# Ian Hay
#
# Decision Tree Implementation 
#
# 2023-01-24
# updated 2023-03-17


import numpy as np
from scipy import stats
import random

import sys
sys.path.append("/Users/ian/Documents/GitHub/ML-Models")
import ML2Utility


class Node():

    def __init__(self, value=None, feature=None, threshold=None, left=None, right=None):
        self.value = value
        self.feature = feature
        self.threshold = threshold
        self.left = left
        self.right = right

    def isLeaf(self):
        return not (self.left or self.right)

    def setLeft(self, leftChild: 'Node'):
        self.left = leftChild

    def setRight(self, rightChild: 'Node'):
        self.right = rightChild

    def printNode(self):
        if self.value is not None:
            print(f"Node.\nValue: {self.value}\n")
        else:
            print(f"Node.\nFeature: {self.feature}\tThreshold: {self.threshold}\n")


class DecisionTree():

    def __init__(self, maxDepth=5, maxSplits=10, minSamples=5, random=False, regression=False, epsilon=0.5):
        self.maximumDepth = maxDepth
        self.maximumSplits = maxSplits
        self.minSamples = minSamples
        self.regression = regression
        self.random = random
        self.depth = None
        self.numSplits = None
        self.root = None
        self.epsilon = epsilon


    def splitNode(self, data, feature, threshold):
        leftIdx = data[feature] <= threshold
        rightIdx = data[feature] > threshold
        return leftIdx, rightIdx


    def bestSplit(self, data, features, label, weights):
        prior = ML2Utility.entropy(data, label, weights)
        n = len(data)

        if (weights is None):
            weights = np.ones((n)) / n

        splitDict = {"Feature": None, "Threshold": None, "Metric": -1}
        if (self.regression):
            splitDict["Metric"] = np.inf

        for feature in features:
            X = data[feature]
            for value in set(X):

                leftIdx, rightIdx = self.splitNode(data, feature, threshold=value)
                leftData, rightData = data[leftIdx], data[rightIdx]
                n_left = len(leftData)
                n_right = len(rightData)
                if (self.regression):
                    if (n_left == 0 or n_right == 0):
                        continue

                    # This regression tree is definitely not working properly.

                    leftPredy = np.full(shape=n_left, fill_value=np.average(leftData[label]), dtype=float)
                    rightPredy = np.full(shape=n_right, fill_value=np.average(rightData[label]), dtype=float)

                    mse = (ML2Utility.mse(leftPredy, leftData[label]) + ML2Utility.mse(rightPredy, rightData[label])) / 2

                    if (mse < splitDict["Metric"]):
                        splitDict["Metric"] = mse
                        splitDict["Feature"] = feature
                        splitDict["Threshold"] = value

                else:
                    leftWeights, rightWeights = weights[leftIdx], weights[rightIdx]
                    childIG = (n_left / float(n)) * ML2Utility.entropy(leftData, label, leftWeights) + (n_right / float(n)) * ML2Utility.entropy(rightData, label, rightWeights)
                    ig = prior - childIG
                
                    if (ig > splitDict["Metric"]):
                        splitDict["Metric"] = ig
                        splitDict["Feature"] = feature
                        splitDict["Threshold"] = value

        return splitDict["Feature"], splitDict["Threshold"], splitDict["Metric"]


    # assumes data is a pandas.DataFrame, features and label are columns of that DataFrame.
    def growTree(self, data, features, label, weights, depth=0, numSplits=0):
        self.depth = depth
        self.numSplits = numSplits
        n = data.shape[0]

        if (self.depth >= self.maximumDepth 
            or self.numSplits >= self.maximumSplits
            or n < self.minSamples
            or len(set(data[label])) <= 1):
            val = -1
            if (self.regression):
                val = np.average(data[label])
            else:
                mode, count = stats.mode(data[label], keepdims=True)
                val = mode[0]
            return Node(value=val)
        if (self.random):
            # select random feature, threshold
            feature = random.choice(features)
            datapoints = list(set(data[feature]))
            threshold = random.choice(datapoints)
            ig = np.inf
        else:
            feature, threshold, ig = self.bestSplit(data, features, label, weights)

        if (self.regression and ig < self.epsilon):
            val = np.average(data[label])
            return Node(value=val)

        leftIdx, rightIdx = self.splitNode(data, feature, threshold)
        leftData, rightData = data[leftIdx], data[rightIdx]
        leftWeights, rightWeights = weights[leftIdx], weights[rightIdx]
        self.numSplits = self.numSplits + 1
        leftChild = self.growTree(leftData, features, label, leftWeights, depth=depth+1, numSplits=self.numSplits)
        rightChild = self.growTree(rightData, features, label, rightWeights, depth=depth+1, numSplits=self.numSplits)
        
        return Node(feature=feature, threshold=threshold, left=leftChild, right=rightChild)


    def makePrediction(self, data, node: 'Node'):
        if (node.isLeaf()):
            return node.value

        if data[node.feature] > node.threshold:
            return self.makePrediction(data, node.right)
        else:
            return self.makePrediction(data, node.left)


    def train(self, data, features, label, weights=None):
        if (weights is None): weights = np.ones(data[features].shape[0]) / data[features].shape[0]
        self.root = self.growTree(data, features, label, weights)
#         self.root.printNode()
#         self.root.left.printNode()
#         self.root.right.printNode()

    def test(self, testing_data, features, label):
        x = testing_data[features]
        predy = [self.makePrediction(xi[1], self.root) for xi in x.iterrows()]
        return np.array(predy)
