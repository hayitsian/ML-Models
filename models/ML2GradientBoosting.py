# Ian Hay
#
# 2023-03-23

import copy
import numpy as np
from scipy.stats import mode
from sklearn.tree import DecisionTreeRegressor


class GradientBoosting():


    def __init__(self, numEstimators, maxDepth=2):
        self.numEstimators = numEstimators
        self.maxDepth = maxDepth
        self.estimators = []

    def train(self, data, features, label):

        x = data[features].values
        y = data[label].values

        y_i = copy.deepcopy(y)

        for i in range(self.numEstimators):

            # train the base estimator
            _estimator = DecisionTreeRegressor(max_depth=self.maxDepth)
            _estimator.fit(x, y_i)
            self.estimators.append(_estimator)

            # update labels for each datapoint accoring to the residual
            predy = _estimator.predict(x)
            y_i -= predy

    def test(self, data, features, label):
        x = data[features].values
        numLabels = x.shape[0]
        predy = np.zeros(numLabels)

        # for each estimator
        for i in range(self.numEstimators):
            # take the sum of the estimator's predictions
            predy += self.estimators[i].predict(x)

        return predy
