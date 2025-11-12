# Ian Hay
#
# KDE Implementation
#
# 2023-04-23

import numpy as np


class KDE():

    def __init__(self, sigma=None):
        self.sigma = sigma

    def kernel(self, x, z):
        return np.exp(-1*np.linalg.norm(x-z)**2)

    
    def predict(self, x_i):
        # for a single test point
        # for each class
        classesDict = dict.fromkeys(self.classes, 0)

        n=0
        for _class in self.classes:
            thisK = 0.
            trainData = self.splitX[n]
            # 1) sum kernel of each train point in class with x_i
            for trainPoint in trainData:
                kern = self.kernel(trainPoint, x_i)
                thisK += kern

            classesDict[_class] = thisK
            n += 1

        # predict class with highest prob of (1) / (2)
        total = sum(classesDict.values(), 0.0)
        classesDict = {k: v / total for k, v in classesDict.items()}

        return min(classesDict, key=classesDict.get)

    def train(self, x:np.array, y:np.array):
        self.X = x
        self.Y = y

        self.classes = list(set(y.flatten()))

        sorter = np.argsort(self.Y)
        self.sortedY = self.Y[sorter]
        
        splitter = np.where(self.sortedY[:-1] != self.sortedY[1:])
        self.splitX = np.split(self.X[sorter], splitter[0] + 1) # https://stackoverflow.com/questions/35822299/partition-training-data-by-class-in-numpy


    def test(self, x):
        return [self.predict(x_i) for x_i in x]
