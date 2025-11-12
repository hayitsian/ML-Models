# Ian Hay - DS4420 HW2
# 2023-01-29

import numpy as np
import pandas as pd
pd.options.mode.chained_assignment = None  # default='warn'
import copy
from scipy import stats

import sys
sys.path.append("/Users/ian/Documents/GitHub/ML-Models/models")
import ML2Utility
import ML2Perceptron as Perceptron
import ML2DecisionTree as DecisionTree
import ML2LinearRegressor as LinearRegressor
import ML2LinearRegressorGD as LinearRegressorGD
import ML2LogisticRegressorGD as LogisticRegressorGD



### ----------------------------------------------------------------------------------------------- ###


def main():

    dfPerceptron = ML2Utility.importData(_filepath="perceptronData.txt")

    perceptronFeatures = [0, 1, 2, 3]
    perceptronLabel = 4

    perceptron = Perceptron.Perceptron(learning_rate=0.8, max_iter=400)

    perceptron.train(dfPerceptron, perceptronFeatures, perceptronLabel)


    ### --------------------------------------------------------------------- ###


    housingColumnNames = ["CRIM", "ZN", "INDUS", "CHAS", "NOX", "RM", "AGE", "DIS", "RAD", "TAX", "PTRATIO", "B", "LSTAT", "MEDV"]
    housingFeatures = ["CRIM", "ZN", "INDUS", "CHAS", "NOX", "RM", "AGE", "DIS", "RAD", "TAX", "PTRATIO", "B", "LSTAT"]
    housingLabel = "MEDV"

    dfHousingTrain = ML2Utility.importData(_filepath="housing_train.txt", _columns=housingColumnNames)
    dfhousingTest = ML2Utility.importData(_filepath="housing_test.txt", _columns=housingColumnNames)
    
    n_housing_train = dfHousingTrain.shape[0]
    n_housing_test = dfhousingTest.shape[0]
    
    dfHousing = pd.concat([dfHousingTrain, dfhousingTest])
    dfHousing = ML2Utility.normalize(_data=dfHousing, _label=housingLabel)

    dfHousingTrainNorm = dfHousing.iloc[0:n_housing_train]
    dfHousingTestNorm = dfHousing.iloc[n_housing_train:]


    ### --------------------------------------------------------------------- ###

    trainHouseBinsLabel = np.histogram(dfHousingTrainNorm[housingLabel], bins=8)[1]
    catTrainHouseData = pd.cut(dfHousingTrainNorm[housingLabel], trainHouseBinsLabel).cat.codes
    catTestHouseData = pd.cut(dfHousingTestNorm[housingLabel], trainHouseBinsLabel).cat.codes


    catLabel = "MEDV CAT"
    dfHousingTrainNorm[catLabel] = catTrainHouseData
    dfHousingTestNorm[catLabel] = catTestHouseData

    dectree = DecisionTree.DecisionTree(regression=False, maxDepth=15, maxSplits=30)

    dectree.train(dfHousingTrainNorm, housingFeatures, catLabel)
    predy = dectree.test(dfHousingTrainNorm, housingFeatures, catLabel)

    predycontinuous = []
    for n in range(len(predy)):
        predycontinuous.append(trainHouseBinsLabel[predy[n]])

    truey = np.array(dfHousingTrainNorm[housingLabel])


    _mse = ML2Utility.mse(predycontinuous, truey)
    print(f"Housing Dataset, Decision Tree (Bucketizing), Training data MSE:\t\t{_mse:0.3f}")

    predy = dectree.test(dfHousingTestNorm, housingFeatures, catLabel)
    predycontinuous = []
    for n in range(len(predy)):
        predycontinuous.append(trainHouseBinsLabel[predy[n]])


    truey = np.array(dfHousingTestNorm[housingLabel])
    _mse = ML2Utility.mse(predycontinuous, truey)
    print(f"Housing Dataset, Decision Tree (Bucketizing), Testing data MSE:\t\t\t{_mse:0.3f}")


    ### --------------------------------------------------------------------- ###


    regtree = DecisionTree.DecisionTree(regression=True, maxDepth=10, maxSplits=20)

    regtree.train(dfHousingTrainNorm, housingFeatures, housingLabel)
    predy = regtree.test(dfHousingTrainNorm, housingFeatures, housingLabel)
    truey = np.array(dfHousingTrainNorm[housingLabel])    

    _mse = ML2Utility.mse(predy, truey)
    print(f"Housing Dataset, Regression Tree, Training data MSE:\t\t\t\t{_mse:0.3f}")

    predy = regtree.test(dfHousingTestNorm, housingFeatures, housingLabel)
    truey = np.array(dfHousingTestNorm[housingLabel])    

    _mse = ML2Utility.mse(predy, truey)
    print(f"Housing Dataset, Regression Tree, Testing data MSE:\t\t\t\t{_mse:0.3f}")

    ### --------------------------------------------------------------------- ###

    linreg = LinearRegressor.LinearRegressor(L2=0)
    linreg.train(dfHousingTrainNorm, housingFeatures, housingLabel)

    predy = linreg.test(dfHousingTrainNorm, housingFeatures, housingLabel)
    _mse = ML2Utility.mse(predy, np.array(dfHousingTrainNorm[housingLabel]))
    print(f"Housing Dataset, Linear Regression (Normal Equations), Training Data MSE:\t{_mse:0.3f}")

    predy = linreg.test(dfHousingTestNorm, housingFeatures, housingLabel)
    _mse = ML2Utility.mse(predy, np.array(dfHousingTestNorm[housingLabel]))
    print(f"Housing Dataset, Linear Regression (Normal Equations), Testing Data MSE:\t{_mse:0.3f}")

    ### --------------------------------------------------------------------- ###

    linreg = LinearRegressor.LinearRegressor(L2=0.5)
    linreg.train(dfHousingTrainNorm, housingFeatures, housingLabel)

    predy = linreg.test(dfHousingTrainNorm, housingFeatures, housingLabel)
    _mse = ML2Utility.mse(predy, np.array(dfHousingTrainNorm[housingLabel]))
    print(f"Housing Dataset, Linear Ridge Regression (Normal Equations), Training Data MSE:\t{_mse:0.3f}")

    predy = linreg.test(dfHousingTestNorm, housingFeatures, housingLabel)
    _mse = ML2Utility.mse(predy, np.array(dfHousingTestNorm[housingLabel]))
    print(f"Housing Dataset, Linear Ridge Regression (Normal Equations), Testing Data MSE:\t{_mse:0.3f}")


    ### --------------------------------------------------------------------- ###

    linregGD = LinearRegressorGD.LinearRegressorGD(learning_rate = 0.005, minimumDelta = 0.00005, maxCycles=5000, stochastic=True)
    linregGD.train(dfHousingTrainNorm, housingFeatures, housingLabel)

    predy = linregGD.test(dfHousingTrainNorm, housingFeatures, housingLabel)
    _mse = ML2Utility.mse(predy, np.array(dfHousingTrainNorm[housingLabel]))
    print(f"Housing Dataset, Linear Regression (Gradient Descent), Training Data MSE:\t{_mse:0.3f}")

    predy = linregGD.test(dfHousingTestNorm, housingFeatures, housingLabel)
    _mse = ML2Utility.mse(predy, np.array(dfHousingTestNorm[housingLabel]))
    print(f"Housing Dataset, Linear Regression (Gradient Descent), Testing Data MSE:\t{_mse:0.3f}")


    ### --------------------------------------------------------------------- ###


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

    dfSpam = ML2Utility.importData(_filepath="spambase.data", _columns=spamColumns, _sep=",")
    
    spamFeatures = copy.deepcopy(spamColumns)
    spamFeatures.remove("spam")
    spamLabel = "spam"

    dfSpamNorm = ML2Utility.normalize(dfSpam, _label=spamLabel)

    _k = 10

    print("\n")

    logregGD = LogisticRegressorGD.LogisticRegressorGD(learning_rate = 0.005, minimumDelta = 0.00005, maxCycles=5000, stochastic=True)
    ML2Utility.rocCurveAucCalculation(dfSpamNorm, logregGD, spamFeatures, spamLabel, k=4, plotTitle="Logistic Regression, Gradient Descent ROC")

    linreg = LinearRegressor.LinearRegressor(L2=0.)
    ML2Utility.rocCurveAucCalculation(dfSpamNorm, linreg, spamFeatures, spamLabel, k=4, plotTitle="Linear Regression, Normal Equations ROC")

    linregGD = LinearRegressorGD.LinearRegressorGD(learning_rate = 0.0005, minimumDelta = 0.00000005, maxCycles=500000, stochastic=True)
    ML2Utility.rocCurveAucCalculation(dfSpamNorm, linregGD, spamFeatures, spamLabel, k=4, plotTitle="Linear Regression, Gradient Descent ROC")

    print("\n")


    ### --------------------------------------------------------------------- ###


    linreg = LinearRegressor.LinearRegressor(L2=0)

    _stats = ML2Utility.kfoldcrossvalidation(dfSpamNorm, linreg, spamFeatures, spamLabel, k=_k, classifier=True, thresh=0.5)
    _tp = 0
    _tn = 0
    _fp = 0
    _fn = 0
    totalN = 0
    for n in range(_k):
        tp, tn, fp, fn = _stats[n]
        _tp = _tp + tp
        _tn = _tn + tn
        _fp = _fp + fp
        _fn = _fn + fn
        totalN = totalN + tn + tp + fp + fn
    _acc = (_tp + _tn) / totalN    
    print(f"Spambase Dataset, Linear Regression (Normal Equations) {_k}-Fold CV Accuracy:\t{_acc:0.3f}")

    _stats = None


    ### --------------------------------------------------------------------- ###


    linreg = LinearRegressor.LinearRegressor(L2=0.5)

    # tp, tn, fp, fn
    _stats = ML2Utility.kfoldcrossvalidation(dfSpamNorm, linreg, spamFeatures, spamLabel, k=_k, classifier=True, thresh=0.5)
    _tp = 0
    _tn = 0
    _fp = 0
    _fn = 0
    totalN = 0
    for n in range(_k):
        tp, tn, fp, fn = _stats[n]
        _tp = _tp + tp
        _tn = _tn + tn
        _fp = _fp + fp
        _fn = _fn + fn
        totalN = totalN + tn + tp + fp + fn
    _acc = (_tp + _tn) / totalN
    print(f"Spambase Dataset, Linear Ridge Regression (Normal Equations), {_k}-Fold CV Accuracy:\t{_acc:0.3f}\n")    

    _stats = None


    ### --------------------------------------------------------------------- ###


    dectree = DecisionTree.DecisionTree(regression=False)

    _stats = ML2Utility.kfoldcrossvalidation(dfSpamNorm, dectree, spamFeatures, spamLabel, k=_k, classifier=True, thresh=None)
    _tp = 0
    _tn = 0
    _fp = 0
    _fn = 0
    totalN = 0
    for n in range(_k):
        tp, tn, fp, fn = _stats[n]
        _tp = _tp + tp
        _tn = _tn + tn
        _fp = _fp + fp
        _fn = _fn + fn
        totalN = totalN + tn + tp + fp + fn
    _acc = (_tp + _tn) / totalN
    print(f"Spambase Dataset, Decision Tree, {_k}-Fold CV Accuracy:\t{_acc:0.3f}")

    ML2Utility.printConfusionMatrix(_tp, _tn, _fp, _fn)

    _stats = None


    ### --------------------------------------------------------------------- ###


    linregGD = LinearRegressorGD.LinearRegressorGD(learning_rate = 0.0005, minimumDelta = 0.00000005, maxCycles=500000, stochastic=True)

    _stats = ML2Utility.kfoldcrossvalidation(dfSpamNorm, linregGD, spamFeatures, spamLabel, k=_k, classifier=True, thresh=0.5)
    _tp = 0
    _tn = 0
    _fp = 0
    _fn = 0
    totalN = 0
    for n in range(_k):
        tp, tn, fp, fn = _stats[n]
        _tp = _tp + tp
        _tn = _tn + tn
        _fp = _fp + fp
        _fn = _fn + fn
        totalN = totalN + tn + tp + fp + fn
    _acc = (_tp + _tn) / totalN

    print(f"Spambase Dataset, Linear Regression (Gradient Descent), {_k}-Fold CV Accuracy:\t{_acc:0.3f}")    

    ML2Utility.printConfusionMatrix(_tp, _tn, _fp, _fn)

    _stats = None


    ### --------------------------------------------------------------------- ###


    logregGD = LogisticRegressorGD.LogisticRegressorGD(learning_rate = 0.005, minimumDelta = 0.00005, maxCycles=5000, stochastic=True)

    _stats = ML2Utility.kfoldcrossvalidation(dfSpamNorm, logregGD, spamFeatures, spamLabel, k=_k, classifier=True, thresh=0.5)
    _tp = 0
    _tn = 0
    _fp = 0
    _fn = 0
    totalN = 0
    for n in range(_k):
        tp, tn, fp, fn = _stats[n]
        _tp = _tp + tp
        _tn = _tn + tn
        _fp = _fp + fp
        _fn = _fn + fn
        totalN = totalN + tn + tp + fp + fn
    _acc = (_tp + _tn) / totalN

    print(f"Spambase Dataset, Logistic Regression (Gradient Descent), {_k}-Fold CV Accuracy:\t{_acc:0.3f}")    

    ML2Utility.printConfusionMatrix(_tp, _tn, _fp, _fn)

    _stats = None


    ### --------------------------------------------------------------------- ###


main()
