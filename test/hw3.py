# Ian Hay - DS4420 HW3
#
# 2023-02-28

import copy
import numpy as np
import pandas as pd
import sys
sys.path.append("/Users/ian/Documents/GitHub/Spring-23/DS4420 ML2")
import ML2Utility
import ML2GaussianDiscriminantAnalysis as GDA
import ML2ExpectationMaximization as EM
import ML2NaiveBayes as NB
np.warnings.filterwarnings('ignore', category=np.VisibleDeprecationWarning)
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.naive_bayes import BernoulliNB, GaussianNB
from sklearn.mixture import GaussianMixture as GMM


def main():

    print("Gaussian Discriminant Analysis, Spambase data...\n")

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

    x = dfSpamNorm[spamFeatures].values
    y = dfSpamNorm[spamLabel].values

    gda = GDA.GDA()

    _stats = ML2Utility.kfoldcrossvalidation(dfSpamNorm, gda, spamFeatures, spamLabel, k=_k, classifier=True, thresh=0.5)
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
    print(f"Spambase Dataset, GDA {_k}-Fold CV Accuracy:\t{_acc:0.3f}")

    lda = LDA(solver='lsqr')
    lda.fit(x, y)
    predy = lda.predict(x)
    print(f"SKLearn GDA Implementation Acc: {ML2Utility.accuracy(predy, y):.3f}")

    
    # want the discrete data for NB
    x = dfSpam[spamFeatures].values
    y = dfSpam[spamLabel].values

    _NB = NB.NB(_class="gaussian")

    _NB.train(dfSpam, spamFeatures, spamLabel)
    predy = _NB.test(dfSpam, spamFeatures, spamLabel)

    _stats = ML2Utility.kfoldcrossvalidation(dfSpam, _NB, spamFeatures, spamLabel, k=_k, classifier=True, thresh=0.5)
    _tp = 0
    _tn = 0
    _fp = 0
    _fn = 0
    totalN = 0
    bigStatDict = {}
    for n in range(_k):
        statDict = {}
        # statDict["Fold"] = [n]
        tp, tn, fp, fn = _stats[n]
        _tp = _tp + tp
        _tn = _tn + tn
        _fp = _fp + fp
        _fn = _fn + fn
        totalN = totalN + tn + tp + fp + fn
        statDict["TP"] = [tp]
        statDict["TN"] = [tn]
        statDict["FP"] = [fp]
        statDict["FN"] = [fn]
        bigStatDict[n] = statDict
    statDict = {}
    statDict["TP"] = [_tp / _k]
    statDict["TN"] = [_tn / _k]
    statDict["FP"] = [_fp / _k]
    statDict["FN"] = [_fn / _k]
    bigStatDict["Average"] = statDict
    statDF = pd.DataFrame(bigStatDict)
    
    _acc = (_tp + _tn) / totalN
    print(statDF.transpose())
    print(f"Spambase Dataset, Gaussian NB {_k}-Fold CV Accuracy:\t{_acc:0.3f}")

    ML2Utility.rocCurveAucCalculation(dfSpam, _NB, spamFeatures, spamLabel, k=4, plotTitle="Gaussian Naive Bayes ROC", numThresh=10)


    _NB = NB.NB(_class="bernoulli")

    _stats = ML2Utility.kfoldcrossvalidation(dfSpam, _NB, spamFeatures, spamLabel, k=_k, classifier=True, thresh=0.5)
    _tp = 0
    _tn = 0
    _fp = 0
    _fn = 0
    totalN = 0
    bigStatDict = {}
    for n in range(_k):
        statDict = {}
        # statDict["Fold"] = [n]
        tp, tn, fp, fn = _stats[n]
        _tp = _tp + tp
        _tn = _tn + tn
        _fp = _fp + fp
        _fn = _fn + fn
        totalN = totalN + tn + tp + fp + fn
        statDict["TP"] = [tp]
        statDict["TN"] = [tn]
        statDict["FP"] = [fp]
        statDict["FN"] = [fn]
        bigStatDict[n] = statDict
    statDict = {}
    statDict["TP"] = [_tp / _k]
    statDict["TN"] = [_tn / _k]
    statDict["FP"] = [_fp / _k]
    statDict["FN"] = [_fn / _k]
    bigStatDict["Average"] = statDict
    statDF = pd.DataFrame(bigStatDict)
    
    _acc = (_tp + _tn) / totalN
    print(f"Spambase Dataset, Bernoulli NB {_k}-Fold CV Accuracy:\t{_acc:0.3f}")
    print(statDF.transpose())

    ML2Utility.rocCurveAucCalculation(dfSpam, _NB, spamFeatures, spamLabel, k=4, plotTitle="Bernoulli Naive Bayes ROC")



    gNB = GaussianNB()
    gNB.fit(x, y)
    predy = gNB.predict(x)
    print(f"SKLearn Gaussian NB Implementation Acc: {ML2Utility.accuracy(predy, y):.3f}")

    bNB = BernoulliNB()
    bNB.fit(x, y)
    predy = bNB.predict(x)
    print(f"SKLearn Bernoulli NB Implementation Acc: {ML2Utility.accuracy(predy, y):.3f}")

    #########################################################################################################

    x = np.array(pd.read_csv("2gaussian.txt", header=None, sep=" ").values).astype(float)
    gmm = GMM(n_components=2, verbose=1, init_params='kmeans')
    gmm.fit(x)
    print(f"SKLearn Pi: {gmm.weights_}")
    print(f"SKLearn Mu: {gmm.means_}")
    print(f"SKLearn Cov: {gmm.covariances_}")
    
    _EM = EM.EM(maxIter=50, verbose=False, seed=42, solver="gaussian")
    pi, mu, cov = _EM.train(x, 2)
    print(f"Pi: {pi}\n Mu: {mu}\n Cov: {cov}")


    x = np.array(pd.read_csv("3gaussian.txt", header=None, sep=" ").values).astype(float)
    gmm = GMM(n_components=3, verbose=1, init_params='kmeans')
    gmm.fit(x)
    print(f"SKLearn Pi: {gmm.weights_}")
    print(f"SKLearn Mu: {gmm.means_}")
    print(f"SKLearn Cov: {gmm.covariances_}")
    
    _EM = EM.EM(maxIter=50, verbose=False, seed=42, solver="gaussian")
    pi, mu, cov = _EM.train(x, 3)
    print(f"Pi: {pi}\n MU: {mu}\n Cov: {cov}")


    ########################################################################################

    nClasses = 2
    nFlips = 10
    nSamples = 5000
    _pi = np.array([0.8, 0.2])
    _pr = np.array([0.75, 0.4])

    x = np.zeros(nSamples, dtype=int)
    for i in range(nSamples):
        _coin = np.argmax(np.random.multinomial(1, _pi))
        x[i] = np.random.binomial(nFlips, _pr[_coin])

    _EM = EM.EM(maxIter=25, verbose=False, seed=42, solver="binomial")
    pi, q = _EM.train(x, nClasses, nFlips)
    print(f"True Pi: {_pi}\n True Q: {_pr}")
    print(f"Pi: {pi}\n Q: {q}")


    nClasses = 2
    nFlips = 10
    nSamples = 5000
    _pi = np.array([0.6, 0.4])
    _pr = np.array([0.6, 0.75])

    x = np.zeros(nSamples, dtype=int)
    for i in range(nSamples):
        _coin = np.argmax(np.random.multinomial(1, _pi))
        x[i] = np.random.binomial(nFlips, _pr[_coin])

    _EM = EM.EM(maxIter=50, verbose=False, seed=42, solver="binomial")
    pi, q = _EM.train(x, nClasses, nFlips)
    print(f"True Pi: {_pi}\n True Q: {_pr}")
    print(f"Pi: {pi}\n Q: {q}")


main()