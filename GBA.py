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
from yellowbrick.classifier import PrecisionRecallCurve
from yellowbrick.classifier import ConfusionMatrix


dataFrame = pd.read_csv(r"C:\Users\TIM\Desktop\BCD-Copy.csv")

dataFrame.replace('?', -99999, inplace=True)
columns = dataFrame.columns.tolist()
columns = [c for c in columns if c not in ["Class", "Sample code number"]]
target = "Class"
X = dataFrame[columns]
Y = dataFrame[target]

X_train, X_test, Y_train, Y_test = model_selection.train_test_split(X, Y, test_size = 0.3)  # test ~ x training ~ y


GBA = GradientBoostingClassifier(learning_rate=0.58419,n_estimators=172, max_depth=14, min_samples_split=201, min_samples_leaf=5, subsample=1, random_state=42, max_features=1, warm_start=True)
    # learning_rate = determines the impact of each tree on the final outcome, low values preferred to make tree robust but need more trees (n_estimators) to model all the relations
    # n_estimators = number of sequential trees to be modeled, can cause over-fitting so it should be balanced with learning rate
    # max_depth = maximum depth of a tree, higher depth -> more specific -> over-fitting, only ~ 560 samples so low end number 5
    # min_samples_split = minimum number of samples required in a node to be splitted, ~ 0.5-1% of samples
    # min_samples_leaf = minimum number of samples required in a leaf, ~ a 10th of split
    # subsample =  fraction (%) of observations to be selected for each tree, between 0.7 and 1 are good, strengthening values
    # random_state = random number seed so that same random numbers are generated every time, if not fixed -> different outcomes for subsequent runs
    # max_features = number of features to consider while searching for best split, 30-40% of total features, higher value -> CAN cause over-fitting
    # warm_start = fit additional trees on previous fits of a model, can use it to increase the number of estimators in small steps and test different values without having to run from scratch
kfold = model_selection.KFold(n_splits=5, shuffle=True, random_state=42)
    ## Assess the dataset model
n_scores = cross_val_score(GBA, X_train, Y_train, cv=kfold, scoring='accuracy')
    ## production of the study
# print('Mean Accuracy: %.3f (%.3f)' % (mean(n_scores), std(n_scores)))
GBA.fit(X_train, Y_train)
GBAprediction = GBA.predict(X_test)
# print(accuracy_score(Y_test, prediction))
# print(classification_report(Y_test, prediction))


# ----------------------------------------------------------------------------------------------------------------------------------------------

SVM = SVC(C = 19.0, gamma = 9.200000000000001e-05)
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

# ---------------------------------------------------------------------------------------------------------------------------------------------- Random forest alg


RFM = RandomForestClassifier(max_depth=5, max_features=1, min_samples_leaf=2, min_samples_split=20, n_estimators=38, random_state=42)
RFM.fit(X_train,Y_train)
RFM_scores = cross_val_score(RFM, X_train, Y_train, cv=kfold, scoring='accuracy')

# -----------------------------------------------------------------------------

predicted_label1 = GBA.predict(X_test)
predicted_label2 = SVM.predict(X_test)
predicted_label3 = RFM.predict(X_test)

accuracy_model1 = accuracy_score(Y_test, predicted_label1)
accuracy_model2 = accuracy_score(Y_test, predicted_label2)
accuracy_model3 = accuracy_score(Y_test, predicted_label3)

sns.barplot(x=["GBA", "SVM", "RFA"], y=[accuracy_model1, accuracy_model2, accuracy_model3])

plt.title("Comparison of Model Accuracy")
plt.xlabel("Model")
plt.ylabel("Accuracy")
plt.ylim(ymin = 0.9, ymax = 1)
plt.show()



#----------------------------- show only 0.7 to 1 on y axis, recolour avrg precision to the same colour as respective PR curve, rename the visualisation and name curves 
viz = PrecisionRecallCurve(GBA, fill_area= False) 
viz.fit(X_train, Y_train)
viz.score(X_test, Y_test)

viz2 = PrecisionRecallCurve(SVM, fill_area= False)
viz2.fit(X_train, Y_train)
viz2.score(X_test, Y_test)

viz2 = PrecisionRecallCurve(RFM, fill_area= False)
viz2.fit(X_train, Y_train)
viz2.score(X_test, Y_test)

viz.show()





f1_GBA = f1_score(Y_test, predicted_label1)
f1_SVM = f1_score(Y_test, predicted_label2)
f1_RFM = f1_score(Y_test, predicted_label3)

f1_data = {'model': ['Gradient Boosting', 'SVM', 'Random Forest'], 'f1_score': [f1_GBA, f1_SVM, f1_RFM]}
sns.barplot(x=f1_data['model'], y=f1_data['f1_score']) 
plt.title("Comparison of Models F1 Score")
plt.xlabel("Model")
plt.ylabel("F1 Score")
plt.ylim(ymin = 0.9, ymax = 1)
plt.show()




visualizer = ROCAUC(GBA, classes=["malignant", "benign"], binary = True)
visualizer.fit(X_train, Y_train)
visualizer.score(X_test, Y_test)

visualizer2 = ROCAUC(SVM, classes=["malignant", "benign"], binary = True)
visualizer2.fit(X_train, Y_train)
visualizer2.score(X_test, Y_test)

visualizer3 = ROCAUC(RFM, classes=["malignant", "benign"], binary = True)
visualizer3.fit(X_train, Y_train)
visualizer3.score(X_test, Y_test)
    
visualizer.show()




ConfMat_GBA = ConfusionMatrix(GBA, classes=[0,1])
ConfMat_GBA.score(X_test, Y_test)
ConfMat_GBA.show()

ConfMat_SVM = ConfusionMatrix(SVM, classes=[0,1])
ConfMat_SVM.score(X_test, Y_test)
ConfMat_SVM.show()

ConfMat_RFM = ConfusionMatrix(RFM, classes=[0,1])
ConfMat_RFM.score(X_test, Y_test)
ConfMat_RFM.show()