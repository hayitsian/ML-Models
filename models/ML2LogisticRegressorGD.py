# Ian Hay
#
# Logistic Regression Implementation 
# with Gradient Descent
#
# 2023-01-29

import numpy as np


class LogisticRegressorGD():

    def __init__(self, learning_rate=0.05, maxCycles=50000, minimumDelta = 0.005, stochastic=False, newtons=False, beta=1.):
        self.learning_rate = learning_rate
        self.minimumDelta = minimumDelta
        self.maxCycles = maxCycles
        self.stochastic = stochastic
        self.beta = beta
        self.weights = []

    def makePredition(self, testX):
        return np.dot(testX, self.weights)

    def sigmoidFunction(self, testX):
        return 1. / (1. + np.exp( -1 * self.beta * self.makePredition(testX)))
    
    def newtonDerivative(self):
        return self.makePredition() * (1- self.makePredition()) * self.X

    def newton(self):
        hx = self.makePredition(self.X)
        dhx = self.newtonDerivative()
        if (dhx == 0):
            return np.full(self.X.shape[1], self.minimumDelta)
        return -1 * np.divide(hx, dhx)

    def gradient(self, index):
        return np.dot((self.sigmoidFunction(self.X[index]) - self.Y[index]), np.transpose(self.X[index]))

    def updateWeight(self, error):
        delta = self.learning_rate * error
        self.weights = self.weights - delta
        self.cycles = self.cycles + 1
        return np.sum(delta)

    def train(self, data, features, label):
        X = np.array(data[features])
        self.X = np.hstack((X,np.ones([X.shape[0],1], X.dtype))) # add column of 1's for y-intercept
        self.Y = np.array(data[label])

        self.cycles = 0
        self.weights = np.ones(self.X.shape[1])
        
        while (self.cycles < self.maxCycles):
            gradient = 0
            for n in range(data.shape[0]):
                gradient = gradient + self.gradient(n)
                if (self.stochastic):
                    delta = self.updateWeight(gradient)
                    # if (delta < self.minimumDelta):
                    #     return
                    gradient = 0

            if (not self.stochastic):
                delta = self.updateWeight( (1./data.shape[0]) * gradient)
                # if (delta < self.minimumDelta):
                #     return


    def test(self, testing_data, features, label):
        testX = np.array(testing_data[features])
        testX = np.hstack((testX,np.ones([testX.shape[0],1], testX.dtype))) # add column of 1's for y-intercept
        return self.sigmoidFunction(testX)
