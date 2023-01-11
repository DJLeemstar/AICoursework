import pandas as pd
from pandas.plotting import scatter_matrix
import numpy as np
from numpy import mean
from numpy import std
import seaborn as sns
import matplotlib.pyplot as plt
import xgboost as xgb
from sklearn import model_selection
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
from sklearn.datasets import make_classification
from sklearn.ensemble import GradientBoostingClassifier


dataFrame = pd.read_csv(r"C:\Users\TIM\Desktop\BCD-Copy.csv")
# dataFrame.head()
dataFrame.replace('?', -99999, inplace=True)
# dataFrame.info()
# print(dataFrame.loc[20])
# print(dataFrame.shape)
# print(dataFrame.describe()) !!!!!!!!!!!!!!!!!!!!!!!!
# dataFrame.hist(figsize=(15, 15))
# plt.show()
# scatter_matrix(dataFrame, figsize=(20, 20))  # plots attributes next to each other to understand correlations
# plt.show()
# ----------- some more graphical steps -----------

columns = dataFrame.columns.tolist()
columns = [c for c in columns if c not in ["Class", "Sample code number"]]
target = "Class"
X = dataFrame[columns]
Y = dataFrame[target]
# print(X.shape) !!!!!!!
# print(Y.shape)!!!!!!!!!!
for x in range(20):
    X_train, X_test, Y_train, Y_test = model_selection.train_test_split(X, Y, test_size = 0.2)  # test ~ 140 training ~ 560
# print(X_train.shape, X_test.shape)
# print(Y_train.shape, Y_test.shape)
# corr = dataFrame.corr()
# sns.heatmap(corr, xticklabels=corr.columns, yticklabels=corr.columns)
    if x % 5 == 0:
        print("----------------------")
    model = GradientBoostingClassifier(learning_rate=0.05,n_estimators=72, max_depth=5, min_samples_split=50, min_samples_leaf=5, subsample=0.8, random_state=10, max_features=3, warm_start=True)
    # learning_rate = determines the impact of each tree on the final outcome, low values preferred to make tree robust but need more trees (n_estimators) to model all the relations
    # n_estimators = number of sequential trees to be modeled, can cause over-fitting so it should be balanced with learning rate
    # max_depth = maximum depth of a tree, higher depth -> more specific -> over-fitting, only ~ 560 samples so low end number 5
    # min_samples_split = minimum number of samples required in a node to be splitted, ~ 0.5-1% of samples
    # min_samples_leaf = minimum number of samples required in a leaf, ~ a 10th of split
    # subsample =  fraction (%) of observations to be selected for each tree, between 0.7 and 1 are good, strengthening values
    # random_state = random number seed so that same random numbers are generated every time, if not fixed -> different outcomes for subsequent runs
    # max_features = number of features to consider while searching for best split, 30-40% of total features, higher value -> CAN cause over-fitting
    # warm_start = fit additional trees on previous fits of a model, can use it to increase the number of estimators in small steps and test different values without having to run from scratch
    kfold = model_selection.KFold(n_splits=10, shuffle=True, random_state=5)
    ## Assess the dataset model
    n_scores = cross_val_score(model, X_train, Y_train, cv=kfold, scoring='accuracy')
    ## production of the study
    print('Mean Accuracy: %.3f (%.3f)' % (mean(n_scores), std(n_scores)))
    model.fit(X_train, Y_train)
    prediction = model.predict(X_test)
    # print(accuracy_score(Y_test, prediction))
    # print(classification_report(Y_test, prediction))
