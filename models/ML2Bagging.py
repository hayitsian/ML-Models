# Ian Hay
#
# 2023-03-23

import numpy as np
from scipy.stats import mode
from sklearn.tree import DecisionTreeClassifier


class Bagging():

    def __init__(self, baseEstimator='decision tree', numEstimators=50, numDatapoints=100, maxDepth=5):
        self.numEstimators = numEstimators
        self.baseEstimator = baseEstimator
        self.maxDepth = maxDepth
        self.numDatapoints = numDatapoints
        self.estimators = []

    def train(self, data, features, label):
        yTrue = data[label].values
        x = data[features].values
        numData = len(yTrue)
        if (self.numDatapoints > numData):
            raise ValueError(f"Bagging subset size {self.numDatapoints} is greater than number of datapoints {numData}")

        for i in range(self.numEstimators):
            
            # sample numEstimators points from the dataset (with replacement)
            indices = np.random.choice(list(range(numData)), self.numDatapoints, replace=True)
            x_i = x[indices]
            y_i = yTrue[indices]

            # train the classifier with the subset of data
            _estimator = DecisionTreeClassifier(max_depth=self.maxDepth)
            _estimator.fit(x_i, y_i)
            self.estimators.append(_estimator)


    def test(self, data, features, label):
        x = data[features].values
        predy = []

        for i in range(self.numEstimators):

            # predict x for each estimator
            _preds = self.estimators[i].predict(x)
            predy.append(_preds)

        # take the most frequently predicted class for each point
        return mode(predy, 0, keepdims=True)[0][0]
