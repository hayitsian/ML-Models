# Ian Hay
#
# Linear Regression Implementation 
# with Algebraic Normal Equations
#
# 2023-01-24

import numpy as np


class LinearRegressor():

    def __init__(self, L2=0):
        self.L2 = L2
        pass
    

    def train(self, data, features, label):
        X = np.array(data[features])
        self.X = np.hstack((X,np.ones([X.shape[0],1], X.dtype))) # add column of 1's for y-intercept
        self.Y = np.array(data[label])
        xTx_inv = np.linalg.pinv(np.matmul(np.transpose(self.X), self.X) + self.L2*np.identity(self.X.shape[1]))
        xTy = np.matmul(np.transpose(self.X), self.Y)
        self.W = np.matmul(xTx_inv, xTy)


    def test(self, testing_data, features, label):
        testX = np.array(testing_data[features])
        self.testX = np.hstack((testX,np.ones([testX.shape[0],1], testX.dtype))) # add column of 1's for y-intercept
        self.testY = np.array(testing_data[label])
        self.predY = np.matmul(self.testX, self.W)
        return self.predY
