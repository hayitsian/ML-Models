# Ian Hay
#
# Utility Functions
#
# 2023-01-29

import copy
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import itertools
from scipy.sparse import coo_array
from sortedcontainers import SortedSet
from sklearn.model_selection import KFold


def importData(_filepath: str,  _columns: list = [], _sep: str = "\s+",):
    if (_columns):
        df = pd.read_csv(_filepath, sep=_sep, names=_columns, header=None, index_col=None)
    else:
        df = pd.read_csv(_filepath, sep=_sep, header=None, index_col=None)

    return df


def parser(_filepathConfig: str):
    # assumes _filepathConfig is : filename + ".config"
    with open(_filepathConfig) as config:
        data = config.readlines()
    
    if len(data) < 2:
        raise ValueError(f"Config file does not contain enough lines: {len(data)} lines")

    numDatapoints = data[0].split()[0] # first line is # of datapoints, # discrete, # continuous attributed
    labels = data[1] # second is possible labels
    labelLine = len(data) - 2
    features = data[1:labelLine] # remaining are the features
    for i in range(len(features)):
        features[i] = features[i].replace("\n","")

    _filepathData = _filepathConfig.split(".")[0] + ".data"
    dfraw = importData(_filepathData, _sep="\t")
    dfraw = dfraw.replace({"?": np.NAN})
    for column in dfraw.columns:
        dfraw[column].fillna(dfraw[column].mode()[0], inplace=True)

    print(dfraw.head())

    _data = []

    j = 0
    for line in features:
        linesplit = line.split("\t")
        numAttrs = linesplit[0]
        if (numAttrs == "-1000"):
            _data.append(dfraw[j].to_numpy(dtype=np.float64))
            j += 1
            continue
        elif (numAttrs == "1"):
            break
        else:
            onehot = pd.get_dummies(dfraw[j]) # one-hot encoding
            _data.append(onehot.to_numpy(dtype=np.int8))
        j += 1

    _data.append(dfraw[labelLine-1].to_numpy()) # get the label column without encoding

    df = pd.DataFrame(np.column_stack(_data))
    numColumns = df.shape[1]
    
    return df, numColumns


def importSparse(_filepathConfig: str, _filepathSparse: str):

    with open(_filepathConfig) as config:
        configdata = config.readlines()
    numLabels = configdata[2].split("=")[1]
    numRows = int(configdata[3].split("=")[1])
    numCols = int(configdata[4].split("=")[1])


    with open(_filepathSparse) as file:
        data = file.readlines()

    i = 0
    _data = []
    _row = []
    _col = []
    _labels = []
    for line in data:
        linesplit = line.split()
        features = linesplit[1:]
        for _feat in features:
            feat = _feat.split(":")
            featureCol = int(feat[0])
            featureVal = float(feat[1])
            _data.append(featureVal)
            _col.append(featureCol)
            _row.append(i)
        
        label = int(linesplit[0])
        _data.append(label)
        _col.append(numCols)
        _row.append(i)
        # _labels.append(label)

        i += 1

    return coo_array((np.array(_data), (np.array(_row), np.array(_col)))).toarray(), numCols#, np.array(_labels)


def ECOC(_labels, size=20, posLabel=1, negLabel=0):
    labelSet = list(SortedSet(_labels))
    numLabels = len(labelSet)
    np.random.seed(seed=42)
    ECOClabels = np.where(np.random.rand(numLabels, size) > 0.5, posLabel, negLabel)
    ecocDict = {}
    # print(ECOClabels)
    # print(labelSet)
    for _label in labelSet:
        ecocDict[int(_label)] = ECOClabels[int(_label)]
    return ecocDict


# def VotingSchema(_labels)



def HAARfeatures(xTrain, xTest, width=28, height=28, numRectangles=100, verbose=False):

    size = width*height
    numTrain = xTrain.shape[0]
    numTest = xTest.shape[0]
    xTrainReshape = xTrain.reshape(numTrain, width, height)
    xTestReshape = xTest.reshape(numTest, width, height)

    xTrainHaar = []
    xTestHaar = []

    # for each rectangle:
    for i in range(numRectangles):
        # pick random rectangle (topLeft, bottomRight points)
        cols = np.random.choice(list(range(width)), size=2)
        rows = np.random.choice(list(range(height)), size=2)

        minCol = min(cols)
        maxCol = max(cols)
        minRow = min(rows)
        maxRow = max(rows)

        if (maxCol - minCol < 5):
            if (minCol > 4):
                minCol -= 4
            elif (maxCol < 23):
                maxCol += 4
        if (maxRow - minRow < 5):
            if (minRow > 4):
                minRow -= 4
            elif (maxRow < 23):
                maxRow += 4

        # split horizontally and vertically
        midCol = round((minCol + maxCol) / 2)
        midRow = round((minRow + maxRow) / 2)

        if verbose: print(f"Cols: {minCol}, {midCol}, {maxCol}")
        if verbose: print(f"Rows: {minRow}, {midRow}, {maxRow}")


        # count the black pixels in top left, top right, bottom left, and bottom right
        # subregions
        _tlSumTrain = np.array([np.sum(xTrainReshape[i, minRow:midRow, minCol:midCol]) for i in range(numTrain)])
        _trSumTrain = np.array([np.sum(xTrainReshape[i, minRow:midRow, midCol:maxCol]) for i in range(numTrain)])
        _blSumTrain = np.array([np.sum(xTrainReshape[i, midRow:maxRow, minCol:midCol]) for i in range(numTrain)])
        _brSumTrain = np.array([np.sum(xTrainReshape[i, midRow:maxRow, midCol:maxCol]) for i in range(numTrain)])


        _tlSumTest = np.array([np.sum(xTestReshape[i, minRow:midRow, minCol:midCol]) for i in range(numTest)])
        _trSumTest = np.array([np.sum(xTestReshape[i, minRow:midRow, midCol:maxCol]) for i in range(numTest)])
        _blSumTest = np.array([np.sum(xTestReshape[i, midRow:maxRow, minCol:midCol]) for i in range(numTest)])
        _brSumTest = np.array([np.sum(xTestReshape[i, midRow:maxRow, midCol:maxCol]) for i in range(numTest)])

        # compute the difference between the regions above
        #  to obtain 2 features per rectangle

        # (top left + top right) - (bottom left + bottom right)
        xTrainHaar.append((_tlSumTrain + _trSumTrain) - (_blSumTrain + _brSumTrain))
        xTestHaar.append((_tlSumTest + _trSumTest) - (_blSumTest + _brSumTest))
        # (top left + bottom left) - (top right + bottom right)
        xTrainHaar.append((_trSumTrain + _brSumTrain) - (_tlSumTrain + _blSumTrain))
        xTestHaar.append((_trSumTest + _brSumTest) - (_tlSumTest + _blSumTest))
    
    return np.array(xTrainHaar).T, np.array(xTestHaar).T


# zero mean, unit variance normalization
def normalizeDf(_data: pd.DataFrame, _label: str):
    df_norm = _data.copy()
    features = list(_data.columns)
    features.remove(_label)
    for feature in features:
        featureData = np.array(_data[feature])
        std = np.std(featureData)
        if (std != 0):
            featureData = normalize(featureData)
        df_norm[feature] = featureData
    return df_norm


# zero mean, unit variance normalization
# for numpy array
def normalize(x: np.array):
    return (x - x.mean(axis=0)) / x.std(axis=0)


def mse(predy, testy):
    N = len(testy)
    if (N == 0):
        return 0
    return np.sum(np.square(predy - testy)) / N


def accuracy(predy, testy):
    return np.sum(testy == predy) / len(testy)


# assumes binary 0,1 classes
def classificationStats(predy, testy, pos=1, neg=0):
    t = np.where(predy == testy)
    f = np.where(predy != testy)

    tp = np.abs(np.sum(predy[np.where(predy[t] == pos)]))
    tn = np.abs(np.sum(predy[np.where(predy[t] == neg)]))
    fp = np.abs(np.sum(predy[np.where(predy[f] == pos)]))
    fn = np.abs(np.sum(predy[np.where(predy[f] == neg)]))
    return tp, tn, fp, fn


def printConfusionMatrix(tp, tn, fp, fn):
    print(f"True Positive: {tp}\tFalse Positive: {fp}\n"
           + f"False Negative: {fn}\tTrue Negative: {tn}\n")


def entropy(_data, _label, weights=None):

    labels = _data[_label].values
    classes = set(labels) # either [0, 1, 2, ..] or [(0,1.5), (1.5, 3), etc.]
    classesDict = dict.fromkeys(classes, 0)
    n = len(labels)
    n_classes = len(classes)
    if (weights is None): # same length as data
        weights = np.ones((n)) / float(n)

    if n <= 1:
        return 0
    if n_classes <= 1:
        return 0

    for i in range(n):
        datapoint = labels[i]
        classesDict[datapoint] += 1 * weights[i]

    entropy = 0
    m = np.sum(weights)
    for v in classesDict.values():
        p = v / m # / (m * n)
        entropy += (p * np.log2(p))
    
    return -1 * entropy



def kfoldcrossvalidation(_data, _model, features, label, k=10, classifier=False, thresh=None):
    # np.warnings.filterwarnings('ignore', category=np.VisibleDeprecationWarning)  
    n = len(_data[label])
    indexes = range(n)

    randomizedIndexes = np.random.choice(indexes, n, replace=False)
    splitIndexes = np.array_split(randomizedIndexes, k)

    i = 0
    statlist = []
    for testIndices in splitIndexes:
        trainIndices = copy.deepcopy(splitIndexes)
        trainIndices = np.delete(trainIndices, i)
        trainIndices = list(itertools.chain(*trainIndices))

        testData = _data.iloc[testIndices]
        trainData = _data.iloc[trainIndices]

        _model.train(trainData, features, label)
        predy = np.array(_model.test(testData, features, label))
        truey = np.array(testData[label])

        stats = -1

        if (classifier):
            if (thresh):
                predy = np.where(predy < thresh, 0, 1)
            stats = classificationStats(predy, truey)
        else:
            stats = mse(predy, truey)
        statlist.append(stats)

        i = i + 1

    return statlist


def aucCalc(yPred, yTrue, numThresh=10, upperVal=1, lowerVal=0, upperLimit=1., lowerLimit=0.):
    threshes = np.linspace(upperLimit, lowerLimit, numThresh)
    fpr = []
    tpr = []

    fpr.append(0.)
    tpr.append(0.)

    for _thresh in threshes:
        yPredThresh = np.where(yPred > _thresh, upperVal, lowerVal) 
        tp, tn, fp, fn = classificationStats(yPredThresh, yTrue, pos=upperVal, neg=lowerVal)

        if (tn + fp > 0):
            _fpr = fp / (tn + fp)
            fpr.append(_fpr)
        else:
            fpr.append(0.)

        if (tp + fn > 0):
            _tpr = tp / (tp + fn)
            tpr.append(_tpr)
        else:
            tpr.append(0.)

        
    fpr.append(1.)
    tpr.append(1.)

    _auc = 0.

    for _n in range(min((len(fpr) - 1, len(tpr) - 1))):
        slice = ((tpr[_n] + tpr[_n+1]) / 2) * (fpr[_n+1] - fpr[_n])
        _auc = _auc + slice

    return _auc, fpr, tpr



def rocCurve(yPred, yTrue, numThresh=10, upperVal=1, lowerVal=0, upperLimit=1., lowerLimit=0., plotTitle="ROC Curve"):
    _auc, fpr, tpr = aucCalc(yPred, yTrue,  numThresh=numThresh, upperVal=upperVal, lowerVal=lowerVal, upperLimit=upperLimit, lowerLimit=lowerLimit)

    print(f"{plotTitle} AUC:\t{_auc:.3f}")

    plt.plot(fpr, tpr, label=f"{plotTitle} AUC: {_auc:.3f}")
    plt.plot([0,1], [0,1], label=f"Random: {0.5}")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(plotTitle)
    plt.legend()
    plt.savefig(plotTitle, bbox_inches="tight")
    plt.close()


def activeLearning(model, data, features, label, startingTrainSize=0.05, increment=0.025, testingSize=0.2, finalTrainSize=0.6):
    kfold = KFold(shuffle=True)
    for train, test in kfold.split(data[features].values, data[label].values):
        trainIndices, testIndices = train, test
        break
    initialTrainIndices = np.random.choice(trainIndices, round(startingTrainSize * len(trainIndices)))
    totalTrain = startingTrainSize

    increments = []
    errors = []

    while totalTrain < finalTrainSize:
        increments.append(totalTrain)
        untrain = np.setdiff1d(trainIndices, initialTrainIndices)
        localError, trainError, testError, testAUC = model.train(data.iloc[initialTrainIndices], features, label, testingData=data.iloc[testIndices] ,verbose=False)
        errors.append(testError[-1])
        print(f"Test error for {int(totalTrain * 100)}% data: {testError[-1]:.3f}")
        output = model.test(data.iloc[testIndices], features, label)
        output = np.argsort(np.abs(output), kind='stable') # sorted in increasing confidence
        numToAdd = int(increment * len(trainIndices))
        nextIndices = untrain[output[:numToAdd + 1]]
        initialTrainIndices = np.append(initialTrainIndices, nextIndices, axis=0)
        np.random.shuffle(initialTrainIndices)

        totalTrain += increment
    return increments, errors


def randomLearning(model, data, features, label, startingTrainSize=0.05, increment=0.025, testingSize=0.2, finalTrainSize=0.6):
    kfold = KFold(shuffle=True)
    for train, test in kfold.split(data[features].values, data[label].values):
        trainIndices, testIndices = train, test
        break
    initialTrainIndices = np.random.choice(trainIndices, round(startingTrainSize * len(trainIndices)))
    totalTrain = startingTrainSize

    increments = []
    errors = []

    while totalTrain < finalTrainSize:
        increments.append(totalTrain)
        untrain = np.setdiff1d(trainIndices, initialTrainIndices)
        localError, trainError, testError, testAUC = model.train(data.iloc[initialTrainIndices], features, label, testingData=data.iloc[testIndices] ,verbose=False)
        errors.append(testError[-1])
        print(f"Test error for {int(totalTrain * 100)}% data: {testError[-1]:.3f}")
        output = model.test(data.iloc[testIndices], features, label)
        print(output)
        numToAdd = int(increment * len(trainIndices))
        nextIndices = untrain[:numToAdd + 1]
        initialTrainIndices = np.append(initialTrainIndices, nextIndices, axis=0)
        np.random.shuffle(initialTrainIndices)

        totalTrain += increment

    return increments, errors





# deprecated, this is a terrible way to do it
def rocCurveAucCalculation(_data, _model, features, label, plotTitle, k=10, numThresh=6):
    if (_model._class == "bernoulli"):
        threshes = np.linspace(10., -10., numThresh)
    elif (_model._class == "gaussian"):
        threshes = np.linspace(100., -5., numThresh)
    else:
        threshes = np.linspace(1., 0., numThresh)

    fpr = []
    tpr = []

    fpr.append(0.)
    tpr.append(0.)

    for _thresh in threshes:

        stats = kfoldcrossvalidation(_data, _model, features, label, k=k, classifier=True, thresh=_thresh)
        _tp = 0
        _tn = 0
        _fp = 0
        _fn = 0
        totalN = 0
        for n in range(k):
            tp, tn, fp, fn = stats[n]
            _tp = _tp + tp
            _tn = _tn + tn
            _fp = _fp + fp
            _fn = _fn + fn
            totalN = totalN + tn + tp + fp + fn

        if (_tn + _fp > 0):
            _fpr = _fp / (_tn + _fp)
            fpr.append(_fpr)

        if (_tp + _fn > 0):
            _tpr = _tp / (_tp + _fn)
            tpr.append(_tpr)


    fpr.append(1.)
    tpr.append(1.)

    _auc = 0.


    for _n in range(numThresh + 1):
        slice = ((tpr[_n] + tpr[_n+1]) / 2) * (fpr[_n+1] - fpr[_n])
        _auc = _auc + slice

    print(f"Spambase Dataset, {plotTitle} AUC:\t{_auc:.3f}")

    plt.plot(fpr, tpr, label=f"{plotTitle} AUC: {_auc:.3f}")
    plt.plot([0,1], [0,1], label=f"Random: {0.5}")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(plotTitle)
    plt.legend()
    plt.savefig(plotTitle+".pdf", bbox_inches="tight")
    # plt.close()
