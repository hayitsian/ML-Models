# Ian Hay
#
# Dual Perceptron Implementation
#
# 2023-04-23

import numpy as np
from math import dist

class DualPerceptron():

    def __init__(self, kernel="linear", sigma=None, maxIter=10):
        self._kernel = kernel
        self.sigma = sigma
        self.maxIter = maxIter

    def kernel(self, x, z):
        if self._kernel == "linear":
            return np.dot(x, z)
        elif self._kernel == "RBF":
            res = dist(x, z)
            res *= -self.sigma
            if res == 0.0 or res == -0.0: return 0.0
            # np.exp(res, res)
            return np.exp(res)


    def train(self, x, y):
        self.X = x
        self.Y = y

        # initialize alphas, threshold `b`, weights
        n = self.X.shape[1]
        m = self.X.shape[0]

        self.ms = np.zeros(m) # counts misclassified
        self.b = 0

        if self._kernel == "RBF" and self.sigma is None: self.sigma = 1./n

        nIncorrect = np.inf
        nIter = 0

        while nIncorrect > 0 and nIter < self.maxIter:

            y_p = [np.sign(np.sum([self.ms[i] * self.Y[i] * self.kernel(self.X[i], self.X[j]) for i in range(m)])) for j in range(m)]

            misclassified = np.where(y_p != self.Y, 1, 0)
            
            self.ms += misclassified

            nIncorrect = np.sum(misclassified)
            nIter += 1

            print(f"Iteration {nIter}, # Misclassified: {nIncorrect}")

    def test(self, x):
        preds = [np.sign(np.sum([self.ms[i] * self.Y[i] * self.kernel(self.X[i], x_j) for i in range(self.X.shape[0])])) for x_j in x]
        return preds
