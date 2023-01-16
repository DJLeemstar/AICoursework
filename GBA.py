import pandas as pd
from pandas.plotting import scatter_matrix
import numpy as np
from numpy import mean
from numpy import std
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn import model_selection
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
from sklearn.datasets import make_classification
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, BaggingClassifier
import matplotlib.pyplot as plt
import seaborn as sns
from yellowbrick.classifier import ROCAUC
from sklearn.metrics import f1_score, precision_score, recall_score


dataFrame = pd.read_csv(r"C:\Users\TIM\Desktop\BCD-Copy.csv")

dataFrame.replace('?', -99999, inplace=True)
columns = dataFrame.columns.tolist()
columns = [c for c in columns if c not in ["Class", "Sample code number"]]
target = "Class"
X = dataFrame[columns]
Y = dataFrame[target]

X_train, X_test, Y_train, Y_test = model_selection.train_test_split(X, Y, test_size = 0.2)  # test ~ 140 training ~ 560


GBA = GradientBoostingClassifier(learning_rate=0.05,n_estimators=72, max_depth=5, min_samples_split=50, min_samples_leaf=5, subsample=0.8, random_state=10, max_features=3, warm_start=True)
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
n_scores = cross_val_score(GBA, X_train, Y_train, cv=kfold, scoring='accuracy')
    ## production of the study
# print('Mean Accuracy: %.3f (%.3f)' % (mean(n_scores), std(n_scores)))
GBA.fit(X_train, Y_train)
GBAprediction = GBA.predict(X_test)
# print(accuracy_score(Y_test, prediction))
# print(classification_report(Y_test, prediction))

"""

# Get an iterator that generates predictions after each iteration
staged_predict = GBA.staged_predict(X_test)

# Compute the accuracy at each iteration and store it in a list
accuracies = []
for y_pred in staged_predict:
    accuracy = accuracy_score(Y_test, y_pred)
    accuracies.append(accuracy)

# Plot the accuracy over time
plt.plot(accuracies)
plt.xlabel('Iteration')
plt.ylabel('Accuracy')
plt.title('Gradient Boosting Classifier Accuracy')
plt.show()

"""

# ----------------------------------------------------------------------------------------------------------------------------------------------

SVM = SVC(kernel='rbf', random_state = 1)
SVM.fit(X_train,Y_train)
SVM_scores = cross_val_score(SVM, X_train, Y_train, cv=kfold, scoring='accuracy')

""" # useless plot

accuracies = [accuracy_score(Y_test, SVM.predict(X_test))]

# Plot the accuracy over time
plt.plot(accuracies)
plt.xlabel('Iteration')
plt.ylabel('Accuracy')
plt.title('SVM Classifier Accuracy')
plt.show()

"""

# ----------------------------------------------------------------------------------------------------------------------------------------------
# Random forest alg



# Train the classifier
RFM = RandomForestClassifier(max_depth=5, n_estimators=40)
RFM.fit(X_train,Y_train)
RFM_scores = cross_val_score(RFM, X_train, Y_train, cv=kfold, scoring='accuracy')


def Algs(model1, model2, model3):
    predicted_label1 = model1.predict(X_test)
    predicted_label2 = model2.predict(X_test)
    predicted_label3 = model3.predict(X_test)

    accuracy_model1 = accuracy_score(Y_test, predicted_label1)
    accuracy_model2 = accuracy_score(Y_test, predicted_label2)
    accuracy_model3 = accuracy_score(Y_test, predicted_label3)

# ---------------------------------------------------- accuracy
    # Create a bar plot
    sns.barplot(x=["Model 1", "Model 2", "Model3"], y=[accuracy_model1, accuracy_model2, accuracy_model3])

    # Add a title and labels
    plt.title("Comparison of Model Accuracy")
    plt.xlabel("Model")
    plt.ylabel("Accuracy")
    plt.ylim = (0.8, 1.0)  # not working yet
    plt.show()

# ----------------------------------------------- Precision Recall (either for each or all in same)

    from yellowbrick.classifier import PrecisionRecallCurve

    viz = PrecisionRecallCurve(model1, fill_area= False)
    viz.fit(X_train, Y_train)
    viz.score(X_test, Y_test)

    viz2 = PrecisionRecallCurve(model2, fill_area= False)
    viz2.fit(X_train, Y_train)
    viz2.score(X_test, Y_test)

    viz.show()



# -------------------------------------- f1

    f1_model1 = f1_score(Y_test, predicted_label1)
    f1_model2 = f1_score(Y_test, predicted_label2)
    f1_model3 = f1_score(Y_test, predicted_label3)

    f1_data = {'model': ['Gradient Boosting', 'SVM', 'Random Forest'], 'f1_score': [f1_model1, f1_model2, f1_model3]}  # works kek
    sns.barplot(x=f1_data['model'], y=f1_data['f1_score'])  # perhaps plot precision and recall next to it?
    plt.title("Comparison of Model F1 Score")
    plt.show()

# ---------------------------------------- ROC AUC (only works with binary classification so quite good for us) (perhaps all in one but better to do 3x for readability)

    visualizer = ROCAUC(model1, classes=["malignant", "benign"])

    visualizer.fit(X_train, Y_train)        # Fit the training data to the visualizer
    visualizer.score(X_test, Y_test)        # Evaluate the model on the test data
    visualizer.show()                       # Finalize and show the figure
   
# ------------------------------------------ Confusion Matrix (can't do all in one, so call 3x)
    from yellowbrick.classifier import ConfusionMatrix

    cm = ConfusionMatrix(model1, classes=[0,1])

    cm.score(X_test, Y_test)

    cm.show()



Algs(GBA, SVM, RFM)