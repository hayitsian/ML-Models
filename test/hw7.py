# Ian Hay - DS4420 HW7
#
# 2023-04-23


import copy
import numpy as np
import pandas as pd
import sys
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold
from joblib import Parallel, delayed

sys.path.append("/Users/ian/Documents/GitHub/Spring-23/DS4420 ML2")
import ML2Utility
import ML2KNN
import ML2DualPerceptron
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

xTrain, xTest, yTrain, yTest = train_test_split(x, y, test_size=0.2)

print("Kernel Density Estimation, Spambase data")

knn = ML2KNN.KDE()
knn.train(xTrain, yTrain)
preds = knn.test(xTest)

_acc = ML2Utility.accuracy(preds, yTest)
print(f"Accuracy: {_acc:.3f}")


###############################################################################

print("\nDual perceptron, dot product kernel, linear data")

dfPerceptron = ML2Utility.importData(_filepath="perceptronData.txt")

perceptronFeatures = [0, 1, 2, 3]
perceptronLabel = 4

dfPerceptron = ML2Utility.normalizeDf(dfPerceptron, _label=perceptronLabel)

perceptron = ML2DualPerceptron.DualPerceptron(maxIter=500)

x = np.array(dfPerceptron[perceptronFeatures].values)
y = np.array(dfPerceptron[perceptronLabel].values)

perceptron.train(x, y)

###############################################################################

print("\nDual perceptron, dot product kernel, spiral data")

dfSpiral = ML2Utility.importData(_filepath="spiral.txt")
spiralFeatures = [0, 1]
spiralLabel = 2

dfSpiral = ML2Utility.normalizeDf(dfSpiral, _label=spiralLabel)

perceptron = ML2DualPerceptron.DualPerceptron(maxIter=20)

x = np.array(dfSpiral[spiralFeatures].values)
y = np.array(dfSpiral[spiralLabel].values)


perceptron.train(x, y)

###############################################################################

print("\nDual perceptron, RBF kernel, spiral data")

perceptron = ML2DualPerceptron.DualPerceptron(kernel="RBF", sigma=2.2, maxIter=50)

perceptron.train(x, y)
