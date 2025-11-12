# Ian Hay
#
# Adaboost Wrapper Implementation 
#
# 2023-03-14

import numpy as np
import pandas as pd

import sys
sys.path.append("/Users/ian/Documents/GitHub/ML-Models")
import ML2DecisionTree as DecisionTree
import ML2Utility
import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree
from sklearn.ensemble import AdaBoostClassifier


class SKAdaboost():

    def __init__(self, numEstimators=50, maxDepth=1):
        self.numEstimators = numEstimators
        self.maxDepth = maxDepth
    
    def fit(self, xtrain, ytrain):
        self.model = AdaBoostClassifier(n_estimators=self.numEstimators)
        self.model.fit(xtrain, ytrain)
        return self.model

    def predict(self, xtest):
        return self.model.predict(xtest)

    def fit_predict(self, xtrain, ytrain, xtest):
        self.model = self.fit(xtrain, ytrain)
        preds = self.predict(xtest)
        return preds


class Adaboost():

    def __init__(self, baseEstimator='decision tree', maxDepth=1, random=False, numEstimators=50):
        self.baseEstimator = baseEstimator # one of: decision tree -  can be extended in the future
        self.numEstimators = numEstimators
        self.estimators = []
        self.alphas = []
        self.feats = []
        self.margins = []
        self.maxDepth = maxDepth
        self.random = random

    def marginAnalysis(self, num=10):
        return sorted(range(len(self.margins)), key=lambda i: self.margins[i])[-num:]

    # for sklearn models for bug fixing
    def fit(self, data, features, label, verbose=False):
        x = data[features].values
        y = data[label].values
        feats = list(range(len(features)))
        # initialize the distribution
        m = x.shape[0]
        Dist = np.full((m), 1./m)
        for i in range(self.numEstimators):
            _estimator = DecisionTreeClassifier(max_depth = self.maxDepth)
            _estimator.fit(x, y, sample_weight=Dist)
            preds = _estimator.predict(x)

            # calculate alpha from error
            missclassified = [int(x) for x in (preds != y)]
            _error = np.dot(Dist, missclassified) / np.sum(Dist) # local round error
            missclassified = [x if x==1 else -1 for x in missclassified] # convert 0 to -1
            
            if verbose: print(f"Local round error for estimator {i}: {_error:.3f}")
            _alpha = 0.5 * np.log((1. - _error)/float(_error))

            # update the distribution

            # for correctly classified points (1) or incorrectly (-1)
            Dist = np.multiply(Dist, np.exp([float(x) * _alpha for x in missclassified]))

            # normalize distribution
            Dist = Dist / np.sum(Dist)

            # append model & alpha
            self.estimators.append(_estimator)
            self.alphas.append(_alpha)
            # super jank way to extract the feature chosen
            _feat = tree.export_text(_estimator).split("<=")[0].split("_")[1].strip()

            self.feats.append(int(_feat))

            # compute error & AUC for all estimators at this point
            predTrain = self.predict(data, features, label)
            predTrain = np.where(predTrain > 0, 1, -1)
            trainingError = sum(predTrain != y) / float(len(y))
            if verbose: print(f"Training error for {i} estimators: {trainingError:.3f}")

        # do margin feature analysis
        self.margins = np.zeros(len(feats))

        for __feat in feats:
            # sum all the alphas where this __feat was selected
            __indices = [i for i, e in enumerate(self.feats) if e == __feat]
            __margin = np.sum([self.alphas[x] for x in __indices])
            self.margins[__feat] += __margin
        # divide by sum of alpha
        self.margins /= np.sum(self.alphas)
        return predTrain

    def predict(self, testing_data, features, label):
        predy = []
        x = testing_data[features].values
        y = testing_data[label].values
        output = np.zeros(len(y))
        for j in range(len(self.estimators)):
            output += self.alphas[j] * self.estimators[j].predict(x)

        # predy = np.where(output > 0.5, 1, 0)

        return output


    def train(self, data, features, label, testingData=None, plotting=False, plotTitle="Adaboost", verbose=False):

        # initialize the distribution
        m = data[features].shape[0]
        yTrue = data[label].values
        Dist = np.full((m), 1./m)

        roundError = []
        trainingError = []
        testingError = []
        testingAUC = []

        for i in range(self.numEstimators):
            # train the base estimator
            if (self.baseEstimator == 'decision tree'):
                _estimator = DecisionTree.DecisionTree(maxDepth = self.maxDepth, random=self.random)
            else: raise ValueError(f"Unknown base estimator: {self.baseEstimator}")

            _estimator.train(data, features, label, weights=Dist)

            # get the hypothesis predictions
            preds = _estimator.test(data, features, label)


            # calculate alpha from error
            missclassified = [int(x) for x in (preds != yTrue)]
            _error = np.dot(missclassified, Dist) / np.sum(Dist) # local round error
            roundError.append(_error)
            missclassified = [x if x==1 else -1 for x in missclassified]

            if (verbose): print(f"Local round error for estimator {i}: {_error:.3f}")
            _alpha = 0.5 * np.log((1. - _error)/_error)


            # update the distribution

            # for correctly classified points (1) or incorrectly (-1)
            Dist = np.multiply(Dist, np.exp([float(x) * _alpha for x in missclassified]))

            # normalize distribution
            Dist = Dist / np.sum(Dist)

            # append model & alpha
            self.estimators.append(_estimator)
            self.alphas.append(_alpha)

            # compute error & AUC for all estimators at this point
            predTrain = np.array(self.test(data, features, label))
            predTrain = np.where(predTrain > 0, 1, -1)

            trainingError.append(np.sum(predTrain != yTrue) / float(len(yTrue)))

            if (verbose): print(f"Training error for {i} estimators: {trainingError[i]:.3f}")

            if (testingData is not None):
                yTest = testingData[label]
                predTest = np.array(self.test(testingData, features, label))
                predTestThresh = np.where(predTest > 0, 1, -1)
                testingError.append(np.sum(predTestThresh != yTest) / float(len(yTest)))
                if (verbose): print(f"Testing error for {i} estimators: {testingError[i]:.3f}")

                auc, fpr, tpr = ML2Utility.aucCalc(predTest, yTest, lowerVal=-1, upperLimit=2., lowerLimit=-2.)
                if (verbose): print(f"Testing AUC for {i} estimators: {auc:.3f}")
                testingAUC.append(auc)

        # plotting
        if (plotting and testingData is not None):
            plt.plot(range(self.numEstimators), roundError, label="Local round error")
            plt.plot(range(self.numEstimators), trainingError, label="Training error")
            plt.plot(range(self.numEstimators), testingError, label="Testing error")
            plt.title(plotTitle)
            plt.xlabel("Number of Estimators")
            plt.ylabel("Error")
            plt.legend()
            plt.savefig(plotTitle)
            plt.close()
            plt.plot(range(self.numEstimators), testingAUC, label="Testing AUC")
            plt.title(plotTitle + " AUC")
            plt.xlabel("Number of Estimators")
            plt.ylabel("AUC")
            plt.savefig(plotTitle + " AUC")
            plt.close()
        
        return roundError, trainingError, testingError, testingAUC


    def test(self, testing_data, features, label):
        output = np.zeros(testing_data[features].shape[0])
        for j in range(len(self.estimators)):
            output += self.alphas[j] * self.estimators[j].test(testing_data, features, label)

        return output
    
