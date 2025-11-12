# Ian Hay - DS4420 HW1
# 2023-01-24

import numpy as np
import pandas as pd
import copy
import itertools
from scipy import stats

def importData(_filepath: str,  _columns: list, _sep: str = "\s+",):
    df = pd.read_csv(_filepath, sep=_sep, names=_columns, header=None, index_col=None)
    return df

# zero mean, unit variance normalization
def normalize(_data: pd.DataFrame, _label: str):
    df_norm = _data.copy(deep=True)
    features = list(_data.columns)
    features.remove(_label)
    for feature in features:
        featureData = np.array(_data[feature])
        std = np.std(featureData)
        featureData = featureData - np.mean(featureData)
        featureData = featureData / std
        df_norm[feature] = featureData
    return df_norm


def sse(predy, testy):
    N = len(testy)
    if (N == 0):
        return 0
    return np.sum(np.square(predy - testy)) / N


def accuracy(predy, testy):
    return np.sum(testy == predy) / len(testy)


def entropy(_data, _label):
    labels = _data[_label]
    classes = set(labels) # either [0, 1, 2, ..] or [(0,1.5), (1.5, 3), etc.]
    classesDict = dict.fromkeys(classes, 0)
    n = len(labels)
    n_classes = len(classes)

    if n <= 1:
        return 0
    if n_classes <= 1:
        return 0

    for datapoint in labels:
        classesDict[datapoint] += 1

    entropy = 0
    for v in classesDict.values():
        p = v / n
        entropy = entropy + (p * np.log2(p))
    
    return -1 * entropy



def kfoldcrossvalidation(_data, _model, features, label, k=10, classifier=False, thresh=None):
    np.warnings.filterwarnings('ignore', category=np.VisibleDeprecationWarning)  
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
        predy = _model.test(testData, features, label)
        truey = testData[label]

        stat = -1

        if (classifier):
            if (thresh):
                predy = np.where(predy < thresh, 0, 1)
            stat = accuracy(predy, truey)
        else:
            stat = sse(predy, truey)
        statlist.append(stat)

        i = i + 1

    return np.mean(statlist)

    # split into k partitions

    # loop through k times, selecting 1 partition to test on
        # train model on remaining k-1 partitions
        # test on left out partition
        # save accuracy/MSE data
    # return average accuracy/MSE metric




#########################################################################################


class Node():

    def __init__(self, value=None, feature=None, threshold=None, left=None, right=None):
        self.value = value
        self.feature = feature
        self.threshold = threshold
        self.left = left
        self.right = right

    def isLeaf(self):
        return not (self.left or self.right)

    def setLeft(self, leftChild: 'Node'):
        self.left = leftChild

    def setRight(self, rightChild: 'Node'):
        self.right = rightChild

    def printNode(self):
        print(f"Node.\nValue: {len(self.value)}\n")




class DecisionTree():

    def __init__(self, maxDepth=5, maxSplits=10, minSamples=5, regression=False):
        self.maximumDepth = maxDepth
        self.maximumSplits = maxSplits
        self.minSamples = minSamples
        self.regression = regression
        self.depth = None
        self.numSplits = None
        self.root = None


    def splitNode(self, data, feature, threshold):
        leftData = data[data[feature] <= threshold]
        rightData = data[data[feature] > threshold]
        return leftData, rightData


    def bestSplit(self, data, features, label):
        prior = entropy(data, label)
        n = len(data)
        splitDict = {"Feature": None, "Threshold": None, "Metric": -1}
        if (self.regression):
            splitDict["Metric"] = np.inf

        for feature in features:
            X = data[feature]
            for value in set(X):

                leftData, rightData = self.splitNode(data, feature, threshold=value)
                n_left = len(leftData)
                n_right = len(rightData)
                if (self.regression):
                    if (n_left == 0 or n_right == 0):
                        continue

                    # This regression tree is definitely not working properly.

                    leftPredy = np.full(shape=n_left, fill_value=np.average(leftData[label]), dtype=float)
                    rightPredy = np.full(shape=n_right, fill_value=np.average(rightData[label]), dtype=float)

                    mse = sse(leftPredy, leftData[label]) + sse(rightPredy, rightData[label])

                    if (mse < splitDict["Metric"]):
                        splitDict["Metric"] = mse
                        splitDict["Feature"] = feature
                        splitDict["Threshold"] = value                        

                else:
                    childIG = (n_left / n) * entropy(leftData, label) + (n_right / n) * entropy(rightData, label)
                    ig = prior - childIG
                
                    if (ig > splitDict["Metric"]):
                        splitDict["Metric"] = ig
                        splitDict["Feature"] = feature
                        splitDict["Threshold"] = value

        # print(splitDict["Feature"])
        # print(splitDict["Threshold"])
        # print(splitDict["Metric"])

        return splitDict["Feature"], splitDict["Threshold"], splitDict["Metric"]


    # assumes data is a pandas.DataFrame, features and label are columns of that DataFrame.
    def growTree(self, data, features, label, depth=0, numSplits=0):
        self.depth = depth
        self.numSplits = numSplits
        n = data.shape[0]

        if (self.depth >= self.maximumDepth 
            or self.numSplits >= self.maximumSplits
            or n < self.minSamples
            or len(set(data[label])) <= 1):
            val = -1
            if (self.regression):
                val = np.average(data[label])
            else:
                mode, count = stats.mode(data[label])
                val = mode[0]
            return Node(value=val)

        feature, threshold, ig = self.bestSplit(data, features, label)
        # print(f"Feature: {feature}. Threshold: {threshold}. Information Gain: {ig}")

        leftData, rightData = self.splitNode(data, feature, threshold)
        self.numSplits = self.numSplits + 1
        leftChild = self.growTree(leftData, features, label, depth=depth+1, numSplits=self.numSplits)
        rightChild = self.growTree(rightData, features, label, depth=depth+1, numSplits=self.numSplits)
        
        return Node(feature=feature, threshold=threshold, left=leftChild, right=rightChild)


    def makePrediction(self, data, node: 'Node'):
        if (node.isLeaf()):
            return node.value

        if data[node.feature] > node.threshold:
            return self.makePrediction(data, node.right)
        else:
            return self.makePrediction(data, node.left)


    def train(self, data, features, label):
        self.root = self.growTree(data, features, label)


    def test(self, testing_data, features, label):
        predy = []
        for x in range(testing_data.shape[0]):
            predy.append(self.makePrediction(testing_data.iloc[x], self.root))
        return np.array(predy)



class LinearRegressor():

    def __init__(self):
        pass
    

    def train(self, data, features, label, L2=0):
        X = np.array(data[features])
        self.X = np.hstack((X,np.ones([X.shape[0],1], X.dtype))) # add column of 1's for y-intercept
        self.Y = np.array(data[label])
        xTx_inv = np.linalg.pinv(np.matmul(np.transpose(self.X), self.X) + L2*np.identity(self.X.shape[1]))
        xTy = np.matmul(np.transpose(self.X), self.Y)
        self.W = np.matmul(xTx_inv, xTy)


    def test(self, testing_data, features, label):
        testX = np.array(testing_data[features])
        self.testX = np.hstack((testX,np.ones([testX.shape[0],1], testX.dtype))) # add column of 1's for y-intercept
        self.testY = np.array(testing_data[label])
        self.predY = np.matmul(self.testX, self.W)
        return self.predY



########################################################################################################


def main():

    columnNames = ["CRIM", "ZN", "INDUS", "CHAS", "NOX", "RM", "AGE", "DIS", "RAD", "TAX", "PTRATIO", "B", "LSTAT", "MEDV"]
    features = ["CRIM", "ZN", "INDUS", "CHAS", "NOX", "RM", "AGE", "DIS", "RAD", "TAX", "PTRATIO", "B", "LSTAT"]
    label = "MEDV"

    df_train = importData(_filepath="housing_train.txt", _columns=columnNames)
    df_test = importData(_filepath="housing_test.txt", _columns=columnNames)
    
    df = pd.concat([df_train, df_test])
    df = normalize(_data=df, _label=label)

    df_train_norm = normalize(_data=df_train, _label=label)
    df_test_norm = normalize(_data=df_test, _label=label)

    linreg = LinearRegressor()
    linreg.train(df_train_norm, features, label, L2=0.05)

    predy = linreg.test(df_train_norm, features, label)
    _sse = sse(predy, np.array(df_train_norm[label]))
    print(f"Housing Dataset, Linear Regression, Training Data MSE: {_sse}")

    predy = linreg.test(df_test_norm, features, label)
    _sse = sse(predy, np.array(df_test_norm[label]))
    print(f"Housing Dataset, Linear Regression, Testing Data MSE: {_sse}")

    _k = 10


    trainHouseBinsLabel = np.histogram(df_train_norm[label])[1]
    catTrainHouseData = pd.cut(df_train_norm[label], trainHouseBinsLabel).cat.codes
    catTestHouseData = pd.cut(df_test_norm[label], trainHouseBinsLabel).cat.codes

    catLabel = "MEDV CAT"
    df_train_norm[catLabel] = catTrainHouseData
    df_test_norm[catLabel] = catTestHouseData


    dectree = DecisionTree(regression=False, maxDepth=10, maxSplits=15)

    dectree.train(df_train_norm, features, catLabel)
    predy = dectree.test(df_train_norm, features, catLabel)

    predycontinuous = []
    for n in range(len(predy)):
        predycontinuous.append(trainHouseBinsLabel[predy[n]])

    truey = np.array(df_train_norm[label])

    _mse = sse(predycontinuous, truey)
    print(f"Housing Dataset, Decision Tree, Training data MSE: {_mse}")

    predy = dectree.test(df_test_norm, features, catLabel)
    predycontinuous = []
    for n in range(len(predy)):
        predycontinuous.append(trainHouseBinsLabel[predy[n]])


    truey = np.array(df_test_norm[label])
    _mse = sse(predycontinuous, truey)
    print(f"Housing Dataset, Decision Tree, Testing data MSE: {_mse}")




    spamColumns = [
        "word_freq_make",
        "word_freq_address",
        "word_freq_all",
        "word_freq_3d",
        "word_freq_our",
        "word_freq_over",
        "word_freq_remove",
        "word_freq_internet",
        "word_freq_order",
        "word_freq_mail",
        "word_freq_receive",
        "word_freq_will",
        "word_freq_people",
        "word_freq_report",
        "word_freq_addresses",
        "word_freq_free",
        "word_freq_business",
        "word_freq_email",
        "word_freq_you",
        "word_freq_credit",
        "word_freq_your",
        "word_freq_font",
        "word_freq_000",
        "word_freq_money",
        "word_freq_hp",
        "word_freq_hpl",
        "word_freq_george",
        "word_freq_650",
        "word_freq_lab",
        "word_freq_labs",
        "word_freq_telnet",
        "word_freq_857",
        "word_freq_data",
        "word_freq_415",
        "word_freq_85",
        "word_freq_technology",
        "word_freq_1999",
        "word_freq_parts",
        "word_freq_pm",
        "word_freq_direct",
        "word_freq_cs",
        "word_freq_meeting",
        "word_freq_original",
        "word_freq_project",
        "word_freq_re",
        "word_freq_edu",
        "word_freq_table",
        "word_freq_conference",
        "char_freq_;",
        "char_freq_(",
        "char_freq_[",
        "char_freq_!",
        "char_freq_$",
        "char_freq_#",
        "capital_run_length_average",
        "capital_run_length_longest",
        "capital_run_length_total",
        "spam"
        ]

    
    dectree = DecisionTree(regression=False)

    df_spam = importData(_filepath="spambase.data", _columns=spamColumns, _sep=",")

    spamFeatures = copy.deepcopy(spamColumns)
    spamFeatures.remove("spam")
    spamLabel = "spam"


    _acc = kfoldcrossvalidation(df_spam, linreg, spamFeatures, spamLabel, k=_k, classifier=True, thresh=0.5)
    print(f"Spambase Dataset, Linear Regression, {_k}-Fold CV Accuracy: {_acc}")


    _acc = kfoldcrossvalidation(df_spam, dectree, spamFeatures, spamLabel, k=_k, classifier=True, thresh=None)
    print(f"Spambase Dataset, Decision Tree, {_k}-Fold CV Accuracy: {_acc}")


main()