# Ian Hay
#
# 2023-03-02

import numpy as np
import scipy
import math

class EM():

    def __init__(self, maxIter=100, verbose=False, seed=None, solver="gaussian"):
        self.maxIter = maxIter
        self.converged = None
        self.verbose = verbose
        self.seed = seed
        self.solver = solver # one of "gaussian", "binomial"

    # this has given me all sorts of trouble
    def multivariateNormal(self, x, mu, cov):
        xXmu = np.array(x) - np.array(mu).reshape(1,-1)
        print(xXmu.shape)
        _base = 1. / (np.power(2*np.pi, self.numFeatures/2.) * np.power(np.linalg.det(cov), 0.5))
        # _exp = np.exp(-0.5 * (xXmu @ np.linalg.pinv(cov) @ xXmu.T))
        _exp = np.exp(-0.5 * ((xXmu @ np.linalg.pinv(cov)) @ xXmu.T))
        return _base * _exp
    

    def binomial(self, n, k, q):
        return scipy.special.comb(n, k, exact=True)*(np.power(q,k))*(np.power((1-q),(n-k)))


    def expectation(self, x, k):
        if (self.solver == "gaussian"):
            # compute expectation of x with pi, mu, var
            p = []
            sumP = 0.
            # for each model
            for i in range(k):
                # compute expectation
                p.append(scipy.stats.multivariate_normal.pdf(x, mean=self.mu[i], cov=self.cov[i]))

            p = np.array(p).reshape(k,-1)
            self.llh.append(np.log(np.sum(p)) / self.numSamples)
            _z = (p * self.pi) / np.sum(p * self.pi, axis=0)

        elif (self.solver == "binomial"):
            _z = np.zeros((self.numSamples, k))
            for i in range(self.numSamples):
                
                numeratorVals = np.zeros((k))
                for j in range(k):
                    numeratorVals[j] = self.pi[j] * self.binomial(self.nFlips, x[i], self.q[j])
                denominator = np.sum(numeratorVals)
                _z[i] = numeratorVals / denominator


        else: raise ValueError(f"Invalid solver: {self.solver}. Must be one of 'gaussian', 'binomial'")
        return _z
    

    def maximization(self, x, k, z):

        if (self.solver == "gaussian"):
            # given z
            nk = np.sum(z, axis=1)
            self.pi = np.array(nk / self.numSamples).reshape(-1,1)
            
            # for each model
            for i in range(k):
                # estimate mu
                self.mu[i] = np.array(np.multiply(np.sum(z[i] * x.T, axis=1), 1./nk[i]))
                # estimate covariance
                xMu = np.array(x - self.mu[i])
                self.cov[i] = np.dot((z[i] * xMu.T), xMu) / nk[i]
                self.cov[i] += np.eye(self.numFeatures) * 1e-6 # help with math

        elif (self.solver == "binomial"):

            for i in range(k):
                self.pi[i] = np.sum(z[:,i])/self.numSamples
                self.q[i] = np.sum(z[:,i] * x) / (np.sum(z[:,i])*self.nFlips)


        else: raise ValueError(f"Invalid solver: {self.solver}. Must be one of 'Gaussian', 'Binomial'")


    def train(self, x, k, nFlips=10):
        np.random.seed(self.seed)
        self.converged = False
        self.numSamples = x.shape[0]
        
        self.llh = []
        self.llh.append(-np.inf)

        if (self.solver == "gaussian"):
            self.numFeatures = x.shape[1]
            # initialize pi, mu and var randomly for each class randomly
            self.pi = np.random.uniform(low=0.2, high=0.8, size=k).reshape(-1,1)
            self.mu = [np.random.uniform(low=1.0, high=10.0, size=(self.numFeatures)) for i in range(k)]
            self.cov = [np.random.uniform(low=.5, high=2.3, size=(self.numFeatures, self.numFeatures)) for i in range(k)]
            self.cov = [self.cov[i] @ self.cov[i].T for i in range(k)] # ensure positive deterministic
        elif (self.solver == "binomial"):
            self.nFlips = nFlips
            self.pi = np.ones(k) * (1.0/k)
            self.q = np.random.random_sample(k)/2 + 0.25
        else: raise ValueError(f"Invalid solver: {self.solver}. Must be one of 'Gaussian', 'Binomial'")


        iter = 0
        tolerance = 1e-9
        while (not self.converged and iter < self.maxIter):
            iter += 1

            if self.verbose and self.solver == "gaussian": print(f"Pi: {self.pi}\n Mu: {self.mu}\n Cov: {self.cov}")
            if self.verbose and self.solver == "binomial": print(f"Pi: {self.pi}\n Q: {self.q}")

            # expectation
            z_ = self.expectation(x, k) # array of length k

            # maximization
            self.maximization(x, k, z_) # updates pi, mu, cov

            # check convergence

            # self.converged = self.llh[iter] - self.llh[iter-1] < tolerance * self.llh[iter]
        # print(iter)
        if self.solver == "gaussian": return self.pi, self.mu, self.cov
        if self.solver == "binomial": return self.pi, self.q
