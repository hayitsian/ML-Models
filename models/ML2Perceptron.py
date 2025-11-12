# Ian Hay
#
# Perceptron Implementation
#
# 2023-01-30

import numpy as np

class Perceptron():

    def __init__(self, learning_rate=0.05, epsilon=0.00005, max_iter=10):
        self.learning_rate = learning_rate
        self.epsilon = epsilon
        self.max_iter = max_iter
        self.weights = None
        self.bias = None


    def classifyPoints(self, X):
        return np.where(np.matmul(X, self.weights) + self.bias > 0, 1, -1)

    # returns misclassified indices
    def misclassifiedPoints(self):
        predictedY = self.classifyPoints(self.X)
        return (self.Y != predictedY)

    # assumes linearly separable
    def gradient(self):
        misclassified = self.misclassifiedPoints()
        misclassifiedX = self.X[misclassified]
        misclassifiedY = np.reshape(self.Y, (-1, 1))[misclassified]

        gradient = np.zeros(self.X.shape[1])
        bias = 0

        for n in range(len(misclassifiedX)):
            thisX = misclassifiedX[n]
            thisY = misclassifiedY[n][0]

            bias = bias + thisY
            gradient = gradient + self.learning_rate * thisX * thisY
            
        return gradient, bias

    def train(self, data, features, label):
        delta = [np.inf]
        self.X = np.array(data[features])
        self.Y = np.array(data[label])
        self.weights = np.zeros(self.X.shape[1])
        self.bias = 0

        nIter = 0
        nIncorrect = np.inf

        print("\nPerceptron Training...\n")

        while (nIncorrect > 0
                and nIter < self.max_iter):
            
            misclassified = self.misclassifiedPoints()
            delta, bias = self.gradient()

            self.weights = self.weights + delta
            self.bias = self.bias + bias

            nIter = nIter + 1
            nIncorrect = np.sum(misclassified.astype(int))
            print(f"Iteration {nIter}, # Misclassified: {nIncorrect}")


        print(f"\nClassifier weights: {self.bias} {self.weights}")
        print(f"Normalized weights: {self.weights/(-1 * self.bias)}\n")

    def test(self, data, features, label):
        testX = np.array(data[features])
        return self.classifyPoints(testX)
    
