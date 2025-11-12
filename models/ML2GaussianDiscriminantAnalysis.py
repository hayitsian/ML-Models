# Ian Hay
#
# 2023-02-28

import numpy as np


class GDA():

    def __init__(self):
        pass

    def train(self, data, features, label):
        x = np.array(data[features])
        y = np.array(data[label])

        # calculate prior for each class
        # (assume binary classes {0,1})
        self.prior = np.mean(y) # Bernoulli ?

        # calculate mean of features for each class
        self.mean0 = np.mean(x[y==0], axis=0)
        self.mean1 = np.mean(x[y==1], axis=0)

        # normalize & calculate covariance matrix
        self.x = x.copy()
        self.x[y==0] -= self.mean0
        self.x[y==1] -= self.mean1


        # self.cov = (np.matmul(self.x.T, self.x)) - (np.mean(self.x.T, axis=1) * np.mean(self.x, axis=0))
        self.cov = np.cov(self.x.T)


    def calcProbs(self, x):
        probs = [np.exp(-1. * np.sum((x - self.mean0).dot(np.linalg.pinv(self.cov)) * (x - self.mean0), axis=1)) * (1 - self.prior),
         np.exp(-1. * np.sum((x - self.mean1).dot(np.linalg.pinv(self.cov)) *(x - self.mean1), axis=1)) * (self.prior)]
        return probs

    def test(self, data, features, label):
        x = np.array(data[features])
        # calculate probability p(y|x) for each class
        probs = self.calcProbs(x)
        # take argmax and predict that class
        return np.argmax(probs, axis=0)


