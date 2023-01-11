import numpy as np
import sklearn
from sklearn import model_selection
from sklearn.model_selection import cross_validate
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from pandas.plotting import scatter_matrix
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

data = pd.read_csv("breast-cancer-wisconsin.csv")
data.replace('?', -99999, inplace=True)
print(data.axes)
print(data.columns)
print(data.loc[20])
print(data.shape)
print(data.describe())

columns = data.columns.tolist()
columns = [c for c in columns if c not in ["Class", "ID"]]

target = "Class"

X = data[columns]
y = data[target]

X_train, X_test, y_train, y_test = model_selection.train_test_split(X,y,test_size=0.3)
seed = 5
scoring = 'accuracy'

from warnings import simplefilter
simplefilter(action='ignore',category=FutureWarning)

models = []
models.append(('SVM', SVC(gamma="scale", C=5)))
results = []
names = []

for name, model in models:
    kfold = model_selection.KFold(n_splits=50)
    cv_results = model_selection.cross_val_score(model, X_train, y_train, cv=kfold, scoring=scoring)
    print(cv_results)
    results.append(cv_results)
    names.append(name)
    print("%s: %f (%f)" % (name, cv_results.mean(), cv_results.std()))