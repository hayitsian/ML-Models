# Ian Hay
#
# Support Vector Machine
#
# 2023-04-11

import copy
import numpy as np
from sklearn.base import BaseEstimator

# https://www.ccs.neu.edu/home/vip/teach/MLcourse/6_SVM_kernels/materials/platt-smo-book.pdf

class SVM(BaseEstimator):

    def __init__(self, kernel="linear", C=1.0, tol=0.001, max_passes=5, sigma=None, posLabel=1, negLabel=-1):
        self._kernel = kernel
        self.C = C
        self.tol = tol
        self.max_passes = max_passes
        self.sigma = sigma
        self.posLabel = posLabel
        self.negLabel = negLabel
        self.epsilon = 1e-6

    def fit(self, x, y):
        self.train(x, y)
        return self

    def predict(self, x):
        return self.test(x)
    
    def kernel(self, x, z):
        if self._kernel == "linear":
            return x @ z.T # + self.b
        elif self._kernel == "RBF":
            if self.sigma is None: raise ValueError("If using RBF Kernel, need to pass in value for Sigma.")

            if np.ndim(x) == 1 and np.ndim(z) == 1:
                result = np.exp(-(np.linalg.norm(x - z, 2)) / (2*self.sigma**2))
            elif (np.ndim(x) > 1 and np.ndim(z) == 1) or (np.ndim(x) == 1 and np.ndim(z) > 1):
                result = np.exp(-(np.linalg.norm(x - z, 2, axis=1)) / (2*self.sigma**2))
            elif np.ndim(x) > 1 and np.ndim(z) > 1:
                result = np.exp(-(np.linalg.norm(x[:, np.newaxis] - z[np.newaxis, :], 2, axis=2)) / (2*self.sigma**2))
            return result
        
        else: raise ValueError(f"Invalid kernel type: {self._kernel}")

    def makePrediction(self, xTest):
        return (self.alphas * self.Y) @ self.kernel(self.X, xTest) + self.b
    
    def objective(self):
        return np.sum(self.alphas) - 0.5 * np.sum((self.Y[:, None]*self.Y[None, :]) * self.kernel(self.X,self.X) * (self.alphas[:,None]*self.alphas[None,:]))
    
    def KKT(self, a: np.ndarray):
        """Returns whether KKT conditions are satisfied for a given set Lagrange multipliers _a_. """
        # https://piazza.com/class/lcpo8rrqy246yb/post/165

        h_dist = self.test(self.X) * self.Y
        kkt = np.isclose(h_dist, 1) & (a > 0) & (a < self.C)
        kkt |= (h_dist <= 1) & np.isclose(a, self.C, self.tol)
        kkt |= (h_dist >= 1) & np.isclose(a, 0, self.tol) 
        return kkt # sum(kkt) gives the number of satisfied

    def takeStep(self, index1, index2):

        if index1 == index2: return 0

        a_1 = self.alphas[index1]
        a_2 = self.alphas[index2]
        y_1 = self.Y[index1]
        y_2 = self.Y[index2]
        e_1 = self.E[index1]
        e_2 = self.E[index2]

        s = y_1 * y_2

        old_1 = copy.deepcopy(a_1)
        old_2 = copy.deepcopy(a_2)

        # compute L and H bounds
        if y_1 == y_2:
            L = max(0, old_2 - old_1)
            H = min(self.C, self.C + old_2 - old_1)
        else:
            L = max(0, old_2 + old_1 - self.C)
            H = min(self.C, old_2 + old_1)
        if L == H: return 0

        # compute n
        eta = 2 * self.kernel(self.X[index1], self.X[index2]) - self.kernel(self.X[index1], self.X[index1]) - self.kernel(self.X[index2], self.X[index2])

        if eta >= 0:
            self.alphas[index2] = L
            L_obj = self.objective()
            self.alphas[index2] = H
            H_obj = self.objective()
            self.alphas[index2] = old_2

            if L_obj > H_obj + self.epsilon: a_2 = L
            elif L_obj < H_obj - self.epsilon: a_2 = H
            else: a_2 = old_2
        else:
            a_2 = old_2 - y_2*(e_1 - e_2)/eta
            if (a_2 < L): a_2 = L
            elif (a_2 > H): a_2 = H

        if a_2 < self.epsilon: a_2 = 0
        elif a_2 > self.C - self.epsilon: a_2 = self.C

        if np.abs(a_2 - old_2) < self.epsilon*(a_2 + old_2 + self.epsilon): return 0

        # compute new alpha1
        a_1 = old_1 + s*(old_2 - a_2)

        # compute b_1 and b_2
        b_1 = self.b - e_1 - y_1*(self.alphas[index1] - old_1)*self.kernel(self.X[index1], self.X[index1]) - y_2*(self.alphas[index2] - old_2)*self.kernel(self.X[index1], self.X[index2])
        b_2 = self.b - e_2 - y_1*(self.alphas[index1] - old_1)*self.kernel(self.X[index1], self.X[index2]) - y_2*(self.alphas[index2] - old_2)*self.kernel(self.X[index2], self.X[index2])
        # compute b
        if 0 < self.alphas[index1] < self.C: b_new = b_1
        elif 0 < self.alphas[index2] < self.C: b_new = b_2
        else: b_new = (b_1 + b_2) / 2

        # make the updates to alphas
        self.alphas[index1] = a_1
        self.alphas[index2] = a_2

        # update the errors
        if (a_1 > 0) & (a_1 < self.C): self.E[index1] = 0.0
        if (a_2 > 0) & (a_2 < self.C): self.E[index2] = 0.0

        for i in range(self.X.shape[0]):
            if i != index1 and i != index2:
                self.E[i] += y_1 * (a_1 - old_1)*self.kernel(self.X[index1], self.X[i]) + y_2 * (a_2 - old_2)*self.kernel(self.X[index2], self.X[i]) + b_new - self.b

        # update b
        self.b = b_new

        return 1
    
    def examineExample(self, index2, verbose=False):
        y_2 = self.Y[index2]
        a_2 = self.alphas[index2]
        e_2 = self.E[index2]
        r_2 = e_2 * y_2


        if (r_2 < -self.tol and a_2 < self.C) or (r_2 > self.tol and a_2 > 0):

            if (np.sum(np.where((self.alphas > 0) & (self.alphas < self.C), 1, 0)) > 1):
                # second choice heuristic
                if self.E[index2] > 0: index1 = np.argmin(self.E)
                elif self.E[index2] <= 0: index1 = np.argmax(self.E)

                if self.takeStep(index1, index2): return 1


            nonZorC_alphas = np.where((self.alphas > 0) & (self.alphas < self.C))[0]
            np.random.shuffle(nonZorC_alphas)
            for index1 in nonZorC_alphas:
                if self.takeStep(index1, index2): return 1
            
            shuffledAlphas = list(range(len(self.alphas)))
            np.random.shuffle(shuffledAlphas)
            for index1 in shuffledAlphas:
                if self.takeStep(index1, index2): return 1

        return 0
    
    def train(self, x, y, verbose=False):
        self.X = x
        self.Y = y

        # initialize alphas, threshold `b`, weights
        n = self.X.shape[1]
        m = self.X.shape[0]
        passes = 0
        # self.alphas = np.full(m, self.epsilon)
        self.alphas = np.ones(m)
        self.b = 0

        if self._kernel == "RBF" and self.sigma is None: self.sigma = 1./(n * np.var(self.X, axis=1))

        preds = self.test(self.X)
        self.E = preds - self.Y

        while passes < self.max_passes:

            numChanged = 0
            examineAll = 1

            while numChanged > 0 or examineAll:
                numChanged = 0
                if (examineAll):
                    for i in range(m):
                        numChanged += self.examineExample(i, verbose)
                        # assert np.isclose(np.dot(self.Y, self.alphas), 0.0), "Lagrange multipliers not feasible"
                        numKKT = sum(self.KKT(self.alphas))
                        if np.isclose(numKKT, m, self.tol): return
                        # if verbose: print(f"Number of KKT Satisfied Points: {numKKT}")
                        # if verbose: print(f"Number Changed Alphas: {numChanged}")
                    passes += 1
                    if verbose: print(f"Pass: {passes}")
                else:
                    for i in np.where((self.alphas > 0) & (self.alphas < self.C))[0]:
                        numChanged += self.examineExample(i, verbose)
                        # assert np.isclose(np.dot(self.Y, self.alphas), 0.0), "Lagrange multipliers not feasible"
                        numKKT = sum(self.KKT(self.alphas))
                        if np.isclose(numKKT, m, self.tol): return
                        # if verbose: print(f"Number of KKT Satisfied Points: {numKKT}")
                        # if verbose: print(f"Number Changed Alphas: {numChanged}")
                    passes += 1
                    if verbose: print(f"Pass: {passes}")

                numKKT = sum(self.KKT(self.alphas))

                if verbose: print(f"Number of points & alphas: {m}")
                if verbose: print(f"Number of KKT Satisfied Points: {numKKT}")
                if verbose: print(f"Number Changed Alphas: {numChanged}")

                if (examineAll): examineAll = 0
                elif (numChanged==0): examineAll = 1

                if np.isclose(numKKT, m, self.tol): return

            # passes += 1
            # if verbose: print(f"Pass: {passes}")

    def test(self, xTest):
        return np.where(self.makePrediction(xTest) > 0, self.posLabel, self.negLabel)
    
    def fit_predict(self, xTrain, yTrain, xTest):
        self.train(xTrain, yTrain)
        return self.test(xTest)
