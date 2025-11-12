# Ian Hay - DS4420 HW4b
#
# 2023-03-23

import copy
import numpy as np
import pandas as pd
import sys
sys.path.append("/Users/ian/Documents/GitHub/Spring-23/DS4420 ML2")
import ML2Utility
import ML2Adaboost
import ML2Bagging
import ML2GradientBoosting
from sklearn.model_selection import train_test_split

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

    print("\nBagging SKLearn Decision Trees, Spambase data...\n")

    bagging = ML2Bagging.Bagging()
    bagging.train(dfSpamTrain, spamFeatures, spamLabel)
    preds = bagging.test(dfSpamTest, spamFeatures, spamLabel)
    ML2Utility.rocCurve(preds, dfSpamTest[spamLabel].values, lowerLimit=-1., upperLimit=1., lowerVal=-1, plotTitle="Bagging Decision Trees, Spambase data, SKLearn ROC")


    print("\nAdaBoost SKLearn Decision Trees, Spambase data...\n")

    adaboost = ML2Adaboost.Adaboost(random=False, numEstimators=10)
    adaboost.fit(dfSpamTrain, spamFeatures, spamLabel)
    preds = adaboost.predict(dfSpamTest, spamFeatures, spamLabel)
    ML2Utility.rocCurve(preds, dfSpamTest[spamLabel].values, lowerLimit=-2., upperLimit=2., lowerVal=-1, plotTitle="AdaBoost Decision Trees, Spambase data, SKLearn ROC")

    ###############################################################################

    columnNames = ["CRIM", "ZN", "INDUS", "CHAS", "NOX", "RM", "AGE", "DIS", "RAD", "TAX", "PTRATIO", "B", "LSTAT", "MEDV"]
    features = ["CRIM", "ZN", "INDUS", "CHAS", "NOX", "RM", "AGE", "DIS", "RAD", "TAX", "PTRATIO", "B", "LSTAT"]
    label = "MEDV"

    df_train = ML2Utility.importData(_filepath="housing_train.txt", _columns=columnNames)
    df_test = ML2Utility.importData(_filepath="housing_test.txt", _columns=columnNames)
    
    df = pd.concat([df_train, df_test])
    df = ML2Utility.normalizeDf(_data=df, _label=label)

    df_train_norm = ML2Utility.normalizeDf(_data=df_train, _label=label)
    df_test_norm = ML2Utility.normalizeDf(_data=df_test, _label=label)

    print("\nGradient Boosted SKLearn Decision Trees, Housing data...\n")

    gradientboost = ML2GradientBoosting.GradientBoosting(numEstimators=10)
    gradientboost.train(df_train_norm, features, label)
    predTrain = gradientboost.test(df_train_norm, features, label)
    _mse = ML2Utility.mse(predTrain, df_train_norm[label].values)

    print(f"Training MSE: {_mse:.3f}")

    predTest = gradientboost.test(df_test_norm, features, label)
    _mse = ML2Utility.mse(predTest, df_test_norm[label].values)

    print(f"Testing MSE: {_mse:.3f}")

main()