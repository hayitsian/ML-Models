# Ian Hay - DS4420 HW4a
#
# 2023-03-14

import copy
import math
import numpy as np
import pandas as pd
import sys
sys.path.append("/Users/ian/Documents/GitHub/Spring-23/DS4420 ML2")
import ML2Utility
import ML2Adaboost
import itertools
from sklearn.model_selection import train_test_split
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
import matplotlib.pyplot as plt

def main():


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

    adaboost = ML2Adaboost.Adaboost(random=False, numEstimators=10)
    adaboost.fit(dfSpamTrain, spamFeatures, spamLabel)
    preds = adaboost.predict(dfSpamTest, spamFeatures, spamLabel)
    ML2Utility.rocCurve(preds, dfSpamTest[spamLabel].values, lowerLimit=-2., upperLimit=2., lowerVal=-1, plotTitle="AdaBoost Decision Trees, Spambase data, SKLearn ROC")


    ###############################################################################


    print("\nAdaBoost Decision Trees, Spambase data, best split...\n")

    adaboost = ML2Adaboost.Adaboost(random=False, numEstimators=5)
    adaboost.train(dfSpamTrain, spamFeatures, spamLabel, testingData=dfSpamTest, plotting=True, plotTitle="Adaboost, Spambase data, best split, 10 estimators" ,verbose=True)
    preds = adaboost.test(dfSpamTest, spamFeatures, spamLabel)
    ML2Utility.rocCurve(preds, dfSpamTest[spamLabel].values, lowerLimit=-2., upperLimit=2., lowerVal=-1, plotTitle="AdaBoost Decision Trees, Spambase data, best split ROC")


    print("\nAdaBoost Decision Trees, Spambase data, random split...\n")

    adaboost = ML2Adaboost.Adaboost(random=True, numEstimators=50)
    adaboost.train(dfSpamTrain, spamFeatures, spamLabel, testingData=dfSpamTest, plotting=True, plotTitle="Adaboost, Spambase data, random split, 50 estimators" , verbose=True)
    preds = adaboost.test(dfSpamTest, spamFeatures, spamLabel)
    ML2Utility.rocCurve(preds, dfSpamTest[spamLabel].values, lowerLimit=-2., upperLimit=2., lowerVal=-1, plotTitle="AdaBoost Decision Trees, Spambase data, random split ROC")


    ###############################################################################


    dfCRX, numCRXcolumns = ML2Utility.parser("crx/crx.config")

    crxFeats = list(range(numCRXcolumns - 1))
    crxLabel = numCRXcolumns - 1
    dfCRX[crxLabel] = dfCRX[crxLabel].replace({"+": 1, "-": -1})

    dfCRXNorm = ML2Utility.normalizeDf(dfCRX, crxLabel)
    dfCRXtrain, dfCRXtest = train_test_split(dfCRXNorm, train_size = 0.8)

    print("\nAdaBoost Decision Trees, CRX data, best split...\n")

    adaboost = ML2Adaboost.Adaboost(random=False, numEstimators=5)
    adaboost.train(dfCRXtrain, crxFeats, crxLabel, testingData=dfCRXtest, plotting=True, plotTitle=f"AdaBoost Decision Trees, CRX data, best split", verbose=True)


    ###############################################################################


    dfVOTE, numVOTEcolumns = ML2Utility.parser("vote/vote.config")

    voteFeats = list(range(numVOTEcolumns - 1))
    voteLabel = numVOTEcolumns - 1
    dfVOTE[voteLabel] = dfVOTE[voteLabel].replace({"d": 1, "r": -1})

    dfVOTENorm = ML2Utility.normalizeDf(dfVOTE, voteLabel)
    dfVOTEtrain, dfVOTEtest = train_test_split(dfVOTENorm, train_size = 0.8)

    print("\nAdaBoost Decision Trees, VOTE data, best split...\n")

    adaboost = ML2Adaboost.Adaboost(random=False, numEstimators=5)
    adaboost.train(dfVOTEtrain, voteFeats, voteLabel, testingData=dfVOTEtest, plotting=True, plotTitle=f"AdaBoost Decision Trees, VOTE data, best split", verbose=True)
    

    #################################################################################
                    ### Active Learning versus Random Splitting ###
    #################################################################################


    print("\nAdaBoost Decision Trees, Spambase data, random learning...\n")
    adaboost = ML2Adaboost.Adaboost(random=False, numEstimators=5)
    _incr, _err = ML2Utility.randomLearning(adaboost, dfSpamNorm, spamFeatures, spamLabel, increment=0.05)
    plt.plot(_incr * 100, _err, label="Spambase Random Learning Testing error")

    print("\nAdaBoost Decision Trees, Spambase data, active learning...\n")
    plotTitle = "AdaBoost Decision Trees, Spambase data, active learning vs random learning"
    adaboost = ML2Adaboost.Adaboost(random=False, numEstimators=5)
    _incr, _err = ML2Utility.activeLearning(adaboost, dfSpamNorm, spamFeatures, spamLabel, increment=0.05)
    plt.plot(_incr * 100, _err, label="Spambase Active Learning Testing error")


    plt.title(plotTitle)
    plt.xlabel("% Training Data")
    plt.ylabel("Error")
    plt.legend()
    plt.savefig(plotTitle)
    plt.close()

    ###

    print("\nAdaBoost Decision Trees, CRX data, active learning...\n")
    plotTitle = "AdaBoost Decision Trees, CRX data, active learning vs random learning"
    adaboost = ML2Adaboost.Adaboost(random=False, numEstimators=3)
    _incr, _err = ML2Utility.activeLearning(adaboost, dfCRXNorm, crxFeats, crxLabel, increment=0.05)
    plt.plot(_incr * 100, _err, label="CRX Active Learning Testing error")

    print("\nAdaBoost Decision Trees, CRX data, random learning...\n")
    adaboost = ML2Adaboost.Adaboost(random=False, numEstimators=3)
    _incr, _err = ML2Utility.randomLearning(adaboost, dfCRXNorm, crxFeats, crxLabel, increment=0.05)
    plt.plot(_incr * 100, _err, label="CRX Random Learning Testing error")

    plt.title(plotTitle)
    plt.xlabel("% Training Data")
    plt.ylabel("Error")
    plt.legend()
    plt.savefig(plotTitle)
    plt.close()

    ###

    print("\nAdaBoost Decision Trees, VOTE data, active learning...\n")
    plotTitle = "AdaBoost Decision Trees, VOTE data, active learning vs random learning"
    adaboost = ML2Adaboost.Adaboost(random=False, numEstimators=3)
    _incr, _err = ML2Utility.activeLearning(adaboost, dfVOTENorm, voteFeats, voteLabel, increment=0.05)
    plt.plot(_incr * 100, _err, label="VOTE Active Learning Testing error")

    print("\nAdaBoost Decision Trees, VOTE data, random learning...\n")
    adaboost = ML2Adaboost.Adaboost(random=False, numEstimators=3)
    _incr, _err = ML2Utility.randomLearning(adaboost, dfVOTENorm, voteFeats, voteLabel, increment=0.05)
    plt.plot(_incr * 100, _err, label="VOTE Random Learning Testing error")

    plt.title(plotTitle)
    plt.xlabel("% Training Data")
    plt.ylabel("Error")
    plt.legend()
    plt.savefig(plotTitle)
    plt.close()


    #################################################################################

    #################################################################################

                    ######  Error Correcting Output Codes  ######

    #################################################################################

    #################################################################################


    nEst = 50
    print(f"\n\n20(8) Newsgroups Data, ECOC, Adaboost with {nEst} estimators...\n")

    newsDataTrain, trainLabel = ML2Utility.importSparse("8newsgroup/train.trec/config.txt", "8newsgroup/train.trec/feature_matrix.txt")
    newsDataTrain = ML2Utility.normalizeDf(pd.DataFrame(newsDataTrain), trainLabel)

    ECOCdict = ML2Utility.ECOC(pd.DataFrame(newsDataTrain)[trainLabel])

    y = np.array(
        [ECOCdict[i] for i in newsDataTrain[trainLabel]]
    )
    newsDataTrain = newsDataTrain.drop([trainLabel], axis=1)
    x = newsDataTrain

    newsDataTest, testLabel = ML2Utility.importSparse("8newsgroup/test.trec/config.txt", "8newsgroup/test.trec/feature_matrix.txt")
    newsDataTest = ML2Utility.normalizeDf(pd.DataFrame(newsDataTest), testLabel)

    ytest = newsDataTest[testLabel]
    newsDataTest = newsDataTest.drop([testLabel], axis=1)
    xtest = newsDataTest


    preds = []
    for i in range(y.shape[1]):
        print(f"Learning bit: {i}")
        y_i = y[:,i]
        model = AdaBoostClassifier(n_estimators=nEst)
        predTrain_i = model.fit(x, y_i)

        pred_i = model.predict(xtest)
        preds.append(pred_i)


    preds = np.transpose(preds)

    ypred = []
    for _pred in preds:
        distDict = {}
        for key, val in ECOCdict.items():
             # euclidean distance
            distDict[key] = np.linalg.norm(_pred - val)

        minDist = min(distDict.values())

        ypred.append([key for key in distDict if distDict[key] == minDist][0])


    _acc = np.sum(ypred == ytest) / len(ytest)
    print(f"Testing accuracy: {_acc}\n\n")


main()