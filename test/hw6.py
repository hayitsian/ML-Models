# Ian Hay - DS4420 HW6
#
# 2023-04-11


import copy
import numpy as np
import pandas as pd
import sys
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold
from sklearn.svm import SVC
from sklearn.multiclass import OneVsOneClassifier
from joblib import Parallel, delayed
from collections import Counter
from itertools import combinations

sys.path.append("/Users/ian/Documents/GitHub/Spring-23/DS4420 ML2")
import ML2Utility
import ML2SVM
import mnist


###############################################################################

###############################################################################


spamColumns = [
    "word_freq_make","word_freq_address","word_freq_all","word_freq_3d","word_freq_our","word_freq_over","word_freq_remove","word_freq_internet",
    "word_freq_order","word_freq_mail","word_freq_receive","word_freq_will","word_freq_people","word_freq_report","word_freq_addresses","word_freq_free",
    "word_freq_business","word_freq_email","word_freq_you","word_freq_credit","word_freq_your","word_freq_font","word_freq_000","word_freq_money",
    "word_freq_hp","word_freq_hpl","word_freq_george","word_freq_650","word_freq_lab","word_freq_labs","word_freq_telnet","word_freq_857","word_freq_data",
    "word_freq_415","word_freq_85","word_freq_technology","word_freq_1999","word_freq_parts","word_freq_pm","word_freq_direct","word_freq_cs",
    "word_freq_meeting","word_freq_original","word_freq_project","word_freq_re","word_freq_edu","word_freq_table","word_freq_conference","char_freq_;",
    "char_freq_(","char_freq_[","char_freq_!","char_freq_$","char_freq_#","capital_run_length_average","capital_run_length_longest","capital_run_length_total","spam"
    ]

dfSpam = ML2Utility.importData(_filepath="spambase.data", _columns=spamColumns, _sep=",")

spamFeatures = copy.deepcopy(spamColumns)
spamFeatures.remove("spam")
spamLabel = "spam"

dfSpamNorm = ML2Utility.normalizeDf(dfSpam, _label=spamLabel)
dfSpamNorm[spamLabel] = np.where(dfSpamNorm[spamLabel] == 0, -1, 1)
dfSpamTrain, dfSpamTest = train_test_split(dfSpamNorm, test_size=0.2)

x = dfSpamNorm[spamFeatures].values
y = dfSpamNorm[spamLabel].values

kf = StratifiedKFold(n_splits=10)

skLinearSVMAcc = []
skPolySVMAcc = []
skRBFSVMAcc = []
myLinearSVMAcc = []
myRBFSVMAcc = []

for i, (trainIdx, testIdx) in enumerate(kf.split(x, y)):
    xTrain = x[trainIdx]
    yTrain = y[trainIdx]
    xTest = x[testIdx]
    yTest = y[testIdx]

    skSVM = SVC(kernel="linear")
    skSVM.fit(xTrain, yTrain)
    preds = skSVM.predict(xTest)
    _acc = ML2Utility.accuracy(preds, yTest)
    skLinearSVMAcc.append(_acc)

    skSVM = SVC(kernel="poly")
    skSVM.fit(xTrain, yTrain)
    preds = skSVM.predict(xTest)
    _acc = ML2Utility.accuracy(preds, yTest)
    skPolySVMAcc.append(_acc)

    skSVM = SVC(kernel="rbf")
    skSVM.fit(xTrain, yTrain)
    preds = skSVM.predict(xTest)
    _acc = ML2Utility.accuracy(preds, yTest)
    skRBFSVMAcc.append(_acc)

    mySVM = ML2SVM.SVM(kernel="linear", max_passes=1, tol=0.02)
    mySVM.train(xTrain, yTrain, verbose=False)
    preds = mySVM.test(xTest)
    _acc = ML2Utility.accuracy(preds, yTest)
    myLinearSVMAcc.append(_acc)

    """
    mySVM = ML2SVM.SVM(kernel="RBF", max_passes=1, tol=0.02)
    mySVM.train(xTrain, yTrain, verbose=True)
    preds = mySVM.test(np.array(xTest).reshape(-1,1))
    _acc = ML2Utility.accuracy(preds, yTest)
    print(_acc)
    myRBFSVMAcc.append(_acc)
    """

print(f"SkLearn SVM, SMO linear kernel, spambase data 10-fold accuracy: {np.mean(skLinearSVMAcc):.3f}")
print(f"SkLearn SVM, SMO polynomial kernel, spambase data 10-fold accuracy: {np.mean(skPolySVMAcc):.3f}")
print(f"SkLearn SVM, SMO RBF kernel, spambase data 10-fold accuracy: {np.mean(skRBFSVMAcc):.3f}")
print(f"my SVM, SMO linear kernel, spambase data 10-fold accuracy: {np.mean(myLinearSVMAcc):.3f}")
# print(f"my SVM, SMO RBF kernel, spambase data 10-fold accuracy: {np.mean(myRBFSVMAcc):.3f}")


###############################################################################


print("\nMNIST HAAR Feature Extraction...\n")

mnist.init()

x_train, t_train, x_test, t_test = mnist.load()

print(t_train.shape)
print(t_test.shape)
print(x_train.shape)
print(x_test.shape)

xTrain, xTest = ML2Utility.HAARfeatures(x_train, x_test, width=28, height=28, numRectangles=100, verbose=False)

print(xTrain.shape)
print(xTest.shape)

print(f"\nMNIST Data with HAAR features, ECOC...\n")


print("SKLearn SVM, RBF kernel, MNist data...")
skSVM = SVC(kernel="rbf")
skSVM.fit(xTrain, t_train)
preds = skSVM.predict(xTest)
_acc = ML2Utility.accuracy(preds, t_test)
print(f"Accuracy: {_acc:.3f}\n")


###############################################################################

###############################################################################

xTrain = ML2Utility.normalize(xTrain)
xTest = ML2Utility.normalize(xTest)

print("my SVM, SMO linear kernel, MNist data...")

xTrainSmaller, xTestSmallerIgnore, yTrainSmaller, yTestSmaller = train_test_split(xTrain, t_train, train_size=0.2, stratify=t_train)

print(xTrainSmaller.shape)
print(yTrainSmaller.shape)
print(Counter(yTrainSmaller))

# one-versus-one classification
models = {}

print("One-versus-One classification")

# for each combination of 10 choose 2
for i1, i2 in combinations(range(10), 2):
    print(f"Training {i1} vs. {i2}")
    # get the appropriate training datapoints
    xi1 = []
    yi1 = []
    for i in range(len(xTrainSmaller)):
        if yTrainSmaller[i] == i1:
            xi1.append(xTrainSmaller[i])
            yi1.append(1.)
        elif yTrainSmaller[i] == i2:
            xi1.append(xTrainSmaller[i])
            yi1.append(-1.)
    # train the model & store somewhere
    _model = ML2SVM.SVM(tol=0.02, max_passes=1)
    _model.train(np.array(xi1), np.array(yi1))
    models[(i1, i2)] = _model

preds = []
# for each testing datapoint
for i in range(len(xTest)):
    predDict = dict.fromkeys(range(10), 0)
    # for each model get the prediction
    for (i1, i2), _model in models.items():
        pred = _model.test(np.array(xTest[i]))
        if pred == 1.: predDict[i1] += 1
        else: predDict[i2] += 1
    # label with the class with the most wins
    preds.append(max(predDict, key=predDict.get))
    # (if tie between two classes) use direct match to break tie

_acc = ML2Utility.accuracy(preds, t_test)
print(f"Accuracy: {_acc:.3f}\n")


ECOCdict = ML2Utility.ECOC(t_train, size=50, posLabel=1, negLabel=-1)

y_train = np.array(
    [ECOCdict[i] for i in yTrainSmaller]
)
y_test = np.array(
    [ECOCdict[i] for i in t_test]
)


print("ECOC classification")

r = Parallel(n_jobs=16, backend="threading", verbose=50)(delayed(ML2SVM.SVM(max_passes=1).fit_predict)(xTrainSmaller, y_train[:,i], xTest) for i in range(y_train.shape[1]))


preds = np.transpose(np.array(r))

ypred = []
for _pred in preds:
    distDict = {}
    for key, val in ECOCdict.items():
        # euclidean distance
        distDict[key] = np.linalg.norm(_pred - val)

    minDist = min(distDict.values())

    ypred.append([key for key in distDict if distDict[key] == minDist][0])


_acc = ML2Utility.accuracy(ypred, t_test)
print(f"Testing accuracy: {_acc}\n")

