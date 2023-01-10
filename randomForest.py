# from skopt.space import Real, Categorical, Integer
from sklearn.ensemble import RandomForestClassifier
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib import rcParams
from sklearn import model_selection

import math
plt.style.use('ggplot')
rcParams['figure.figsize'] = (12, 6)

dataset = pd.read_csv("breast_cancer.csv")


# preprocessing of data
dataset.replace('?', -99999, inplace=True)
# get all the data into columns
columns = dataset.columns.tolist()
# remove class and id as they have no effect on the prediction
columns = [c for c in columns if c not in ["Class", "ID"]]
# storing the variable we will prredict with
target = 'Class'

X = dataset[columns]
y = dataset[target]
y = y.map({2: 0, 4: 1})
x_train, x_test, y_train, y_test = model_selection.train_test_split(
    X, y, train_size=0.7, random_state=0)

# show the distribution of the cancer type on the dataset

y.value_counts(normalize=True).plot(kind='bar')
plt.title("percentage distribution of the target variables")
plt.xticks(rotation=0)
plt.xlabel('cancer type')
plt.ylabel("percentage of dataset")
# plt.show()

# explore relationships between variables


# heatmap to explore relationships

corrmat = dataset.corr()
hm = sns.heatmap(corrmat,
                 cbar=True,
                 annot=True,
                 square=True,
                 fmt='.2f',
                 annot_kws={'size': 10},
                 yticklabels=dataset.columns,
                 xticklabels=dataset.columns,
                 cmap="Spectral_r")
# plt.show()


# creating the random_forest_model

# creating a x and y dataset for traing and testing the random_forest_model

# testing options


random_forest_model = RandomForestClassifier(max_depth=5, n_estimators=40)


def CV_test_model(model, X, y, seed):
    scoring = ['accuracy', 'precision', 'recall', 'f1']

    results = model_selection.cross_validate(
        estimator=model, X=X, y=y, cv=seed, scoring=scoring, return_train_score=True)
    return results


def print_results(results):
    print("Mean Training Accuracy: ", results['train_accuracy'].mean()*100,
          "\nMean Training Precision: ", results['train_precision'].mean(),
          "\nMean Training Recall: ", results['train_recall'].mean(),
          "\nMean Training F1 Score: ", results['train_f1'].mean(),
          "\nMean Validation Accuracy: ", results['test_accuracy'].mean()*100,
          "\nMean Validation Precision: ", results['test_precision'].mean(),
          "\nMean Validation Recall: ", results['test_recall'].mean(),
          "\nMean Validation F1 Score: ", results['test_f1'].mean()
          )


# print_results(CV_test_model(random_forest_model, X, y, 5))

# --------------------------------optimizaiton----------------------------------


# now using grid search
# change this to your own parameters
param_grid = {
    'n_estimators': np.arange(10, 2000, 200),
    'max_depth': np.arange(1, 100, 10),
    'min_samples_split': np.arange(0.01, 1.0, 0.02),
    'min_samples_leaf': np.arange(0.01, 0.05),
    'max_features': ['auto', 'sqrt', 'log2'],
}
# training the grid search


def grid_search(x_train, y_train, model, param_grid, seed=5):
    '''this will take in your X and y data and a model with your specified parmeters
    seed is number of cross validations. it increases time drastically

    this will create a grid search model and find the best parameters
    it returns the grid results'''
    grid = model_selection.GridSearchCV(
        estimator=model, param_grid=param_grid, cv=seed)
    grid_results = grid.fit(x_train, y_train)
    return grid_results


# temp_results = grid_search(x_train, y_train, random_forest_model, param_grid)
# print('best model parameters are: \n\n', temp_results.best_params_)


def random_grid_search(X, y, model, param_grid, seed=3):
    '''
    imput your split datasets, model and paramiter grid
    seed is the number of cross validations you want to do. in creases run time dramatically

    this will take your model and create a random search grid doing 300 iterations of searches

    it prints the best parameters and best score

    returns the best model of your model you input.
    '''
    random_search = model_selection.RandomizedSearchCV(
        estimator=model, param_distributions=param_grid, n_iter=300, cv=seed, random_state=42, n_jobs=-1)
    random_search.fit(X, y)
    best_model = random_search.best_estimator_
    print(random_search.best_params_)
    print(random_search.best_score_)
    return best_model


random_grid_best_model = random_grid_search(
    X, y, random_forest_model, param_grid)

'''i recommend doing random grid to find your general best parameters and then doing grid search to find the best around those parameters
'''
# modified parameter grid with values around the best params from random search


param_grid = {
    'n_estimators': [1500, 1600,  1700],
    'max_depth': [7, 11, 15],
    'min_samples_split': [0.03, 0.05, 0.07],
    'min_samples_leaf': [0.005, 0.01, 0.015],
    'max_features': ['auto', 'sqrt', 'log2'],
}
grid_search(x_train, y_train, random_forest_model, param_grid)

'''implement needed
random grid search

then grid search with a new params function that is looking at around the best params area

try different datasets and look at the data displays looking for relatinships
use in the funcitons and see if it makes them more accurate'''
