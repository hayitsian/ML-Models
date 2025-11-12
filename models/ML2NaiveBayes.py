# Ian Hay
#
# 2023-03-01

import numpy as np
np.seterr(divide='ignore', invalid='ignore')

class NB():

    # TODO: add functionality to deal with missing values

    def __init__(self, _class="gaussian", smoothing = True):
        self._class = _class
        self.smoothing = smoothing


    def gaussian(self, x, mu, std):
        return np.exp(-((x - mu / (2*std))**2)) / (np.sqrt(2*np.pi)*std)


    def calcParameters(self, x, y):
        # determine the parameters of x for each class
        # based on the type of NB distribution model
        sorter = np.argsort(y)
        sortedY = y[sorter]
        self.classes = list(set(sortedY))
        self.numClasses = len(self.classes)

        if (self._class == "bernoulli"):

            # booleanize data by mean
            self.mean = np.nanmean(x, axis=0)
            boolX = np.where(x > self.mean, 1, 0)

            # split data by class
            splitter, = np.where(sortedY[:-1] != sortedY[1:])
            splitX = np.split(boolX[sorter], splitter + 1) # https://stackoverflow.com/questions/35822299/partition-training-data-by-class-in-numpy

            classStats = {}
            n = 0
            for _classData in splitX:
                # calculate parameters
                countY = np.sum(np.where(sortedY==self.classes[n],1,0))
                prior = countY / self.numSamples
                occurence = np.log(np.nansum(_classData, axis=0) + self.epsilon ) - np.log(countY + (self.epsilon * self.numClasses))
                classStats[self.classes[n]] = {
                    "prior": np.log(prior),
                    "occurences": occurence
                }
                n += 1
            self.parameters = classStats

        elif (self._class == "gaussian"):
            # calculate prior, mean and std for each feature
            # in each classes' data

            # split data by class
            splitter, = np.where(sortedY[:-1] != sortedY[1:])
            splitX = np.split(x[sorter], splitter + 1) # https://stackoverflow.com/questions/35822299/partition-training-data-by-class-in-numpy

            classStats = {}
            n = 0
            for _classData in splitX:
                std = np.std(_classData, axis=0) + self.epsilon
                mean = np.mean(_classData, axis=0)
                prior = np.sum(np.where(sortedY==self.classes[n],1,0)) / self.numSamples
                classStats[self.classes[n]] = {
                    "mean": mean,
                    "std": std,
                    "prior": prior
                }
                n += 1
            self.parameters = classStats
        else: raise ValueError(f"Invalid distribution class: {self._class}")


    def distribution(self, x):
        # calculate the probability x belongs to each class
        numFeatures = x.shape[1]
        probList = []

        if (self._class == "bernoulli"):

            # TODO: skip features that are nan

            # booleanize x
            boolX = np.where(x > self.mean, 1, 0)

            # for each class, calculate prob and add to list
            for _class, _dict in self.parameters.items():
                bernoulli = np.dot(boolX, _dict["occurences"].T)
                prior = _dict["prior"]
                probList.append(np.exp(prior + bernoulli))
        elif (self._class == "gaussian"):
            # for each class, calculate prob and add to list
            for _class, _dict in self.parameters.items():
                gaussian = np.prod(self.gaussian(x, _dict["mean"], _dict["std"]), axis=1)
                prior = _dict["prior"]
                probList.append(prior * gaussian)
        else: raise ValueError(f"Invalid distribution class: {self._class}")
        return np.array(probList)


    def train(self, data, features, label):
        x = np.array(data[features])
        y = np.array(data[label])
        
        self.numSamples = x.shape[0]

        self.epsilon = 0
        if (self.smoothing and self._class == "gaussian"):
            self.epsilon += 1e-9 * np.var(x, axis=0).max()
        elif (self.smoothing and self._class == "bernoulli"):
            self.epsilon = 1
        
        # calculate parameters for each class
        self.calcParameters(x, y)


    def test(self, data, features, label):
        x = np.array(data[features])
        # calculate probability belonging to each class
        probs = self.distribution(x)
        probs = np.where(probs <= 0, 0.0005, probs)

        # return probs
        return np.log(probs[1]/probs[0])
        
