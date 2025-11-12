# Ian Hay - DS4420 HW2B
#
# 2023-02-09

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from itertools import chain


import sys
sys.path.append("/Users/ian/Documents/GitHub/Spring-23/DS4420 ML2")
import ML2Utility
import ML2FFNeuralNetwork

def sse(_p, _q):
    return np.mean(np.power(_p - _q, 2)) / 2

class torchNet(nn.Module):

    def __init__(self, nInput, nHidden, nOutput, maxIter=10000, learningRate=0.05):
        super(torchNet, self).__init__()
        self.layers = nn.Sequential(
                        nn.Linear(nInput, nHidden),
                        nn.Sigmoid(),
                        nn.Linear(nHidden, nOutput),
                        nn.Sigmoid())

        self.maxIter = maxIter
        self.learningRate = learningRate

    def forward(self, x, y=None):
        x = self.layers(x)
        return x

    def backpropagate(self, lossFunc, optimizer, x, y):
        optimizer.zero_grad()
        output = self(x)
        error = lossFunc(output, y)
        error.backward()
        optimizer.step()
        return error
    
    def train(self, x, y):
        x = torch.from_numpy(np.array(x, dtype="float32"))
        y = torch.from_numpy(np.array(y, dtype="float32"))

        loss = nn.MSELoss()
        optimizer = optim.SGD(self.parameters(), lr=self.learningRate)
        nIter = 0

        while (nIter < self.maxIter):
            error = self.backpropagate(loss, optimizer, x, y)
            if (nIter % (self.maxIter / 10) == 0):
                print(f"Iteration: {nIter}  Error: {error}")
            nIter += 1


def main():


    # autoencoder
    n = 8
    X = Y = np.eye(n)

    _maxIter = 2000
    _learningRate = 0.5

    print("Training ML2 FFNN Implementation: Autoencoder...\n")
    ffNN = ML2FFNeuralNetwork.FFNeuralNetwork(maxIter=_maxIter, learningRate=_learningRate, numHiddenNeurons=3)
    ffNN.train(X, Y)

    out = ffNN.test(X)

    print(np.where(out > 0.5, 1, 0))

    print("\n Training PyTorch FFNN Implementation: Autoencoder...\n")
    _maxIter = 200000
    _learningRate = 0.05

    torchNN = torchNet(nInput=8, nHidden=3, nOutput=8, maxIter=_maxIter, learningRate=_learningRate)
    torchNN.train(X, Y)
    out = torchNN.forward(torch.from_numpy(np.array(X, dtype="float32")))

    print(np.where(out.detach().numpy() > 0.5, 1, 0))
    

    # wine classifier
    _maxIter = 10000
    _learningRate = 0.1

    filepath = "train_wine.csv"
    dfWine = ML2Utility.importData(filepath, _sep=",")

    x = np.array(dfWine.iloc[:, 1:])
    x = (x - x.mean(axis=0)) / x.std(axis=0) # zero mean, unit variance normalization

    y = dfWine.iloc[:, [0]]

    y1 = list(chain(*np.where(y == 1, 1, 0)))
    y2 = list(chain(*np.where(y == 2, 1, 0)))
    y3 = list(chain(*np.where(y == 3, 1, 0)))

    newY = np.array([y1, y2, y3]).T

    print("\nTraining PyTorch Implementation: Wine Data...\n")
    torchNN = torchNet(nInput=13, nHidden=20, nOutput=3, maxIter=_maxIter, learningRate=_learningRate)
    torchNN.train(x, newY)
    out = torchNN.forward(torch.from_numpy(np.array(x, dtype="float32")))


    print("\nTraining ML2 FFNN Implementation: Wine Data...\n")
    ffNN = ML2FFNeuralNetwork.FFNeuralNetwork(maxIter=_maxIter, learningRate=_learningRate, numHiddenNeurons=15)
    ffNN.train(x, newY)
    out = ffNN.test(x)

    
    filepath = "test_wine.csv"
    dfWine = ML2Utility.importData(filepath, _sep=",")

    xTest = np.array(dfWine.iloc[:, 1:])
    xTest = (xTest - xTest.mean(axis=0)) / xTest.std(axis=0) # zero mean, unit variance normalization

    yTest = dfWine.iloc[:, [0]]

    y1 = list(chain(*np.where(yTest == 1, 1, 0)))
    y2 = list(chain(*np.where(yTest == 2, 1, 0)))
    y3 = list(chain(*np.where(yTest == 3, 1, 0)))

    newYTest = np.array([y1, y2, y3]).T


    print("\nTesting PyTorch Implementation: Wine Data...\n")
    out = torchNN.forward(torch.from_numpy(np.array(xTest, dtype="float32")))

    print(f"Error: {sse(out.detach().numpy(), newYTest)}\n")


    print("\nTesting ML2 FFNN Implementation: Wine Data...\n")
    out = ffNN.test(xTest, newYTest)

    print(f"Error: {sse(out, newYTest)}\n")

    # (if done) MNIST dataset

main()