# Ian Hay - DS4420 HW5
#
# 2023-03-30


import copy
import numpy as np
import pandas as pd
import sys
sys.path.append("/Users/ian/Documents/GitHub/Spring-23/DS4420 ML2")
import ML2Utility
import ML2Adaboost
import ML2NaiveBayes
import ML2PrincipleComponentAnalysis
from sklearn.model_selection import train_test_split
from sklearn.ensemble import AdaBoostClassifier
from sklearn.decomposition import PCA
from sklearn.naive_bayes import GaussianNB

from joblib import Parallel, delayed

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


print("\nAdaBoost SKLearn Decision Trees, Spambase data...\n")

adaboost = ML2Adaboost.Adaboost(random=False, numEstimators=300)
adaboost.fit(dfSpamTrain, spamFeatures, spamLabel)
preds = adaboost.predict(dfSpamTest, spamFeatures, spamLabel)
preds = np.sign(preds)
_acc = ML2Utility.accuracy(preds, dfSpamTest[spamLabel].values)
print(f"Adaboost Spambase Accuracy: {_acc:.3f}")


top10feats = adaboost.marginAnalysis()
print(f"Top 10 features by margin: {top10feats}")



dfSpamPollutedTrainX = ML2Utility.importData(_filepath="spam_polluted/train_feature.txt")
dfSpamPollutedTrainY = ML2Utility.importData(_filepath="spam_polluted/train_label.txt")

dfSpamPollutedTestX = ML2Utility.importData(_filepath="spam_polluted/test_feature.txt")
dfSpamPollutedTestY = ML2Utility.importData(_filepath="spam_polluted/test_label.txt")

dfSpamPollutedTrain = pd.concat([dfSpamPollutedTrainX, dfSpamPollutedTrainY], axis=1, ignore_index=True)
dfSpamPollutedTest = pd.concat([dfSpamPollutedTestX, dfSpamPollutedTestY], axis=1, ignore_index=True)

spamPollutedFeatures = list(dfSpamPollutedTrainX.columns)
spamPollutedLabel = list(set(dfSpamPollutedTrain.columns) - set(spamPollutedFeatures))[0]


# maybe normalize the data
xTrain = dfSpamPollutedTrain[spamPollutedFeatures].values
yTrain = dfSpamPollutedTrain[spamPollutedLabel].values
xTest = dfSpamPollutedTest[spamPollutedFeatures].values
yTest = dfSpamPollutedTest[spamPollutedLabel].values



print("\nAdaBoost SKLearn Decision Trees, polluted Spambase data...\n")

adaboost = AdaBoostClassifier(n_estimators=300)

adaboost.fit(xTrain, yTrain)
preds = adaboost.predict(xTest)
_acc = ML2Utility.accuracy(preds, yTest)
print(f"Adaboost polluted Spambase Accuracy: {_acc:.3f}")


###############################################################################


print("\nGaussian Naive Bayes, polluted Spambase data...\n")


gnb = GaussianNB()
gnb.fit(xTrain, yTrain)
preds = gnb.predict(xTest)

_acc = ML2Utility.accuracy(preds, yTest)
print(f"Gaussian Naive Bayes polluted Spambase Accuracy: {_acc:.3f}")


print("\nSKLearn PCA, polluted Spambase data...\n")
pca = PCA(n_components=100)

xTrainReduced = pca.fit_transform(xTrain)
xTestReduced = pca.transform(xTest)


print("\nGaussian Naive Bayes, SKLearn PCA reduced polluted Spambase data...\n")

gnb = GaussianNB()
gnb.fit(xTrainReduced, yTrain)
preds = gnb.predict(xTestReduced)

_acc = ML2Utility.accuracy(preds, yTest)
print(f"Gaussian Naive Bayes polluted reduced Spambase Accuracy: {_acc:.3f}")


print("\nMy PCA, polluted Spambase data...\n")

selfPCA = ML2PrincipleComponentAnalysis.PCA(k=100)

xTrainReduced = selfPCA.train(dfSpamPollutedTrain, spamPollutedFeatures, spamPollutedLabel)
xTestReduced = selfPCA.test(dfSpamPollutedTest, spamPollutedFeatures, spamPollutedLabel)


print("\nGaussian Naive Bayes, my PCA reduced polluted Spambase data...\n")


gnb = GaussianNB()
gnb.fit(xTrainReduced, yTrain)
preds = gnb.predict(xTestReduced)

_acc = ML2Utility.accuracy(preds, yTest)
print(f"Gaussian Naive Bayes, my PCA polluted reduced Spambase Accuracy: {_acc:.3f}")



###############################################################################

print("\nBernoulli Naive Bayes, missing Spambase data...\n")


dfSpamMissingTrain = ML2Utility.importData(_filepath="spambase_20_percent_missing_train.txt", _columns=spamColumns, _sep=",")
dfSpamMissingTest = ML2Utility.importData(_filepath="spambase_20_percent_missing_test.txt", _columns=spamColumns, _sep=",")


bnb = ML2NaiveBayes.NB("bernoulli")
bnb.train(dfSpamMissingTrain, spamFeatures, spamLabel)
preds = bnb.test(dfSpamMissingTest, spamFeatures, spamLabel)
preds = np.where(preds > 0, 1, 0)

_acc = ML2Utility.accuracy(preds, dfSpamMissingTest[spamLabel].values)
print(f"Bernoulli Naive Bayes, missing Spambase Accuracy: {_acc:.3f}")


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


nEst = 200
print(f"\nMNIST Data with HAAR features, ECOC, Adaboost with {nEst} estimators...\n")

ECOCdict = ML2Utility.ECOC(t_train, size=50)

y_train = np.array(
    [ECOCdict[i] for i in t_train]
)
y_test = np.array(
    [ECOCdict[i] for i in t_test]
)


r = Parallel(n_jobs=-2, backend="threading", verbose=50)(delayed(ML2Adaboost.SKAdaboost(numEstimators=nEst).fit_predict)(xTrain, y_train[:,i], xTest) for i in range(y_train.shape[1]))


"""
for i in range(y_train.shape[1]):
    print(f"Learning bit: {i}")
    y_i = y_train[:,i]
    model = AdaBoostClassifier(n_estimators=nEst)
    predTrain_i = model.fit(x_train, y_i)

    pred_i = model.predict(x_test)
    preds.append(pred_i)
"""


preds = np.transpose(np.array(r))


ypred = []
for _pred in preds:
    distDict = {}
    for key, val in ECOCdict.items():
        # euclidean distance
        distDict[key] = np.linalg.norm(_pred - val)

    minDist = min(distDict.values())

    ypred.append([key for key in distDict if distDict[key] == minDist][0])


_acc = np.sum(ypred == t_test) / len(t_test)
print(f"Testing accuracy: {_acc}\n")


###############################################################################

###############################################################################

