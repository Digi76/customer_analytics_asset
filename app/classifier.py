import pandas as pd
import numpy as np
import sqlite3
import os
import re
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
#from imblearn.over_sampling import SMOTE
from sklearn.metrics import roc_auc_score
from sklearn.metrics import confusion_matrix
#import seaborn as sn
import statistics
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score
import operator
import warnings
warnings.filterwarnings('ignore')

def get_lift_gain(X_test, y_test, model):
    X_test_copy = X_test.copy()

    X_test_copy = pd.DataFrame(X_test_copy)
    X_test_copy['ACCEPT_INDICATOR'] = list(y_test)
    X_test_copy['ACCEPT_PREDICTED_CONFIDENCE'] = model.predict_proba(X_test)[:,1]

    X_test_copy = X_test_copy.sort_values(by='ACCEPT_PREDICTED_CONFIDENCE', axis=0, ascending=False)
    avg_acceptance_rate = y_test[y_test==1].shape[0]/y_test.shape[0]

    deciles = np.array_split(X_test_copy,10)

    percent_correct = []

    total_positive = X_test_copy[X_test_copy['ACCEPT_INDICATOR']==1].shape[0]

    for i in range(10):
        valueCounts = deciles[i]['ACCEPT_INDICATOR'].value_counts()
        if 1 in valueCounts.index:
            percent_correct.append(valueCounts[1]/total_positive)
        else:
            percent_correct.append(0)

    model_cum_percent_correct = []
    x = 0

    for val in percent_correct:
        x = x + val
        model_cum_percent_correct.append(x)

    baseline = [val/10 for val in range(1,11,1)]


    model_lift = []

    for a,b in zip(model_cum_percent_correct, baseline):
        model_lift.append((a/b)*100)
    return model_cum_percent_correct, model_lift



if __name__ == '__main__':

    path = 'C:/project/DG_thesis/'
    path = 'C:/Users/DebasishGuha/DG/Personal/MTech/Sem4/Project/work/'
    file = 'EU_OFFER_PROPENSITY_VIEW_new.csv'
    scores = []

    model_config = [{"model":"Decision Tree",
                     "parameters":{'max_depth':[10,20,30,40,50,60,70,80,90,100],
                                   'min_impurity_decrease':[1, 0.1, 0.01, 0.001, 0.0001, 0.00001, 0.000001]}},
                    {"model": "Random Forest",
                     "parameters": {'max_depth':[10, 20, 30, 40, 50],
                                    'n_estimators':[10, 20, 30, 40, 50],
                                    'min_impurity_decrease':[1, 0.1, 0.01, 0.001, 0.0001, 0.00001, 0.000001]}},
                    {"model": "Naive Bayes",
                     "parameters": {'var_smoothing':[1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7, 1e-8, 1e-9]}},
                    {"model": "Logistic Regression",
                     "parameters": {'penalty':['l1','l2'], 'C':[1, 0.1, 0.01, 0.001]}},
                    {"model": "Neural Network",
                     "parameters": {'hidden_layer_sizes':[(10, 1), (100, 1)], 'learning_rate':['constant', 'invscaling', 'adaptive']}}
                    ]



    offer_propensity = pd.read_csv(path + file)

    offer_propensity = offer_propensity.drop(['ACCEPT_PREDICTED', 'ACCEPT_PROPENSITY'],
                                               axis=1)
    offer_propensity = offer_propensity.round(3)

    offer_propensity = offer_propensity.drop(['CUSTOMER_ID', 'OFFER_ID','BUILDING_AGE'],
                                               axis=1)
    data_cleaned = pd.get_dummies(offer_propensity)

    target = data_cleaned['ACCEPT_INDICATOR']
    data_cleaned = data_cleaned.drop('ACCEPT_INDICATOR', axis=1)

    X_train, X_test, y_train, y_test = train_test_split(data_cleaned, target, test_size=0.2, random_state=42)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)


    for model in model_config:
        if model.get('model') == 'Decision Tree':
            model_run = DecisionTreeClassifier()
        if model.get('model') == 'Random Forest':
            model_run = RandomForestClassifier()
        if model.get('model') == 'Naive Bayes':
            model_run = GaussianNB()
        if model.get('model') == 'Logistic Regression':
            model_run = LogisticRegression(solver='liblinear', max_iter=1000)
        if model.get('model') == 'Neural Network':
            model_run = MLPClassifier()

        print('Running a',model.get('model'), ' Model :\n\n')
    #parameters = {'max_depth':[d for d in range(10,110,10)], 'min_impurity_decrease':[1, 0.1, 0.01, 0.001, 0.0001, 0.00001, 0.000001]}
    #decision_tree = DecisionTreeClassifier()
    #grid = GridSearchCV(decision_tree, parameters, cv=5, scoring='roc_auc')
        parameters = model.get('parameters')
        grid = GridSearchCV(model_run, parameters, cv=5, scoring='roc_auc')
        grid.fit(X_train,y_train)

        #decision_tree = grid.best_estimator_
        #print(decision_tree)
        model_run = grid.best_estimator_
        print(model_run)

        # decision_tree_score = roc_auc_score(y_test, decision_tree.predict_proba(X_test)[:,1])
        # print('ROC_AUC score for Decision Tree:', decision_tree_score)
        # scores.append(decision_tree_score)
        #
        # a, b = get_lift_gain(X_test, y_test, decision_tree)

        model_score = roc_auc_score(y_test, model_run.predict_proba(X_test)[:,1])
        print('ROC_AUC score for Decision Tree:', model_score)
        scores.append(model_score)


        a, b = get_lift_gain(X_test, y_test, model_run)

        print(a)