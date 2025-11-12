# Ian Hay
#
# 2023-03-30


import numpy as np

class PCA():


    def __init__(self, k=10):
        self.k = k


    def train(self, df, features, label):
        xTrain = np.array(df[features].values)

        # compute mean & cov for training data
        xMean = np.mean(xTrain, axis=0)
        self.means = xMean
        # subtract mean from cov
        xNormed = xTrain - self.means
        xCov = np.cov(xNormed.T)

        # compute eigenvectors & eigenvalues of covariance matrix
        eigenVal, eigenVec = np.linalg.eigh(xCov)
        # sort by eigenVal
        sortedIdx = np.argsort(eigenVal)[::-1]
        eigenVal = eigenVal[sortedIdx]
        eigenVec = eigenVec[:,sortedIdx]
        print(eigenVec.shape)

        # select `k` largest eigenvectors by sorting eigenvalues
        eigenVecK = eigenVec[:,0:self.k]
        self.eigenVecs = np.array(eigenVecK)
        print(self.eigenVecs.shape)

        # return components as columns of a matrix
        return np.dot(xNormed, self.eigenVecs)


    def test(self, df, features, label):
        xTest = np.array(df[features].values)
        xTestNormed = xTest - self.means
        return np.dot(xTestNormed, self.eigenVecs)
