# Ian Hay
#
# 2023-02-09

import numpy as np

class FFNeuralNetwork():

    # only has 1 hidden layer for ease of implementation
    def __init__(self, numHiddenNeurons=3, learningRate = 0.05, maxIter=10000, lossFunc="sse"):
        self.numHiddenNeurons = numHiddenNeurons
        self.learningRate = learningRate
        self.maxIter = maxIter
        self.lossFunc = lossFunc


    def crossEntropyLoss(self, _p, _q):
        # add some check for _q (e.g., if _q is zero)
        return -1. * (_p * np.log(_q) - (1-_p) * np.log(1-_q))
    
    def L1loss(self, _p, _q):
        return _p - _q

    def sse(self, _p, _q):
        return np.mean(np.power(_p - _q, 2)) / 2

    def sigmoid(self, _x):
        return 1. / (1 + np.exp(-1 * _x))

    # assumes you are inputing the output from the sigmoid function (above)
    def sigmoidDerivative(self, _x):
        return (_x) * (1 - (_x))
    

    def calculateOutputs(self):
        inputLayerOut = self.X
        hiddenLayerOut = np.hstack((self.sigmoid(np.dot(inputLayerOut, self.hiddenWeights)), np.ones((self.X.shape[0], 1)))) # adds 1's to account for bias
        outputLayerOut = self.sigmoid(np.dot(hiddenLayerOut, self.outputWeights))
        return inputLayerOut, hiddenLayerOut, outputLayerOut


    def error(self, predy, testy):
        if (self.lossFunc == "sse"):
            return self.sse(predy, testy)
        elif (self.lossFunc == "cel"):
            return self.crossEntropyLoss(predy, testy)
        elif (self.lossFunc == "L1"):
            return self.L1loss(predy, testy)
        else:
            raise ValueError(f"Function has unknown lossFunc: {self.lossFunc} ")


    def train(self, X, Y):
        self.X = np.array(X)
        self.X = np.hstack((self.X,np.ones([self.X.shape[0],1], self.X.dtype))) # adds 1's to account for bias
        self.Y = np.array(Y)
        # initialize weights and biases of network

        # use some gaussian (zero mean) initialization for sigmoid function
        # weights = numHidden * numInput + numHidden * numOutput
        # +1 to account for bias terms
        self.hiddenWeights = np.random.uniform(low=-0.3, high=0.3, size=(self.X.shape[1], self.numHiddenNeurons))
        self.outputWeights = np.random.uniform(low=-0.3, high=0.3, size=(self.numHiddenNeurons + 1, self.Y.shape[1]))

        # while not done
        numIter = 0
        n = self.X.shape[0]

        while (numIter < self.maxIter):

            # USE MATRICES INSTEAD OF FOR LOOPS

            # for each training datapoint
            # calculate its output in each layer
            inputLayerOut, hiddenLayerOut, outputLayerOut = self.calculateOutputs()

            ##### ------ backpropagation ----- #####
            
            # for each unit in output layer
            # compute the error
            output_error = self.error(outputLayerOut, self.Y)
            if (numIter % (self.maxIter / 10) == 0):
                print(f"Iteration: {numIter}  Error: {output_error}")

            # compute the gradients & update weights
            loss = self.L1loss(outputLayerOut, self.Y) * self.sigmoidDerivative(outputLayerOut)
            gradOut = hiddenLayerOut.T @ (loss)
            self.outputWeights += - self.learningRate * gradOut
            

            # for each unit in hidden layers
            # compute the error with respect to the next highest layer
            # (e.g., for a network with 1 hidden layer, compute error w.r.t. output layer)
            gradHidden = self.X.T @ ((loss @ self.outputWeights.T) * self.sigmoidDerivative(hiddenLayerOut))
            self.hiddenWeights += - self.learningRate * gradHidden[:,:-1]

            # increment
            numIter += 1


    def test(self, X, Y=None):
        self.X = np.array(X)
        self.X = np.hstack((self.X,np.ones([self.X.shape[0],1], self.X.dtype))) # adds 1's to account for bias
        inputLayerOut, hiddenLayerOut, outputLayerOut = self.calculateOutputs()
        return outputLayerOut
