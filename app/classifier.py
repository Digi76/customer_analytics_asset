import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, roc_curve, precision_recall_curve, auc
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import MinMaxScaler
import datetime
from sklearn import metrics
import seaborn as sns

import warnings
import json
warnings.filterwarnings('ignore')

cfg_file_nm = "../config/classifier_config.json"
with open(cfg_file_nm, "r") as f:
    cfg_dict = json.load(f)

model_config = cfg_dict.get('model_config')
file_nm = cfg_dict.get('input_file')
target_var = cfg_dict.get('target_var')

def get_lift_gain(X_test, y_test, model):
    X_test_copy = X_test.copy()

    X_test_copy = pd.DataFrame(X_test_copy)
    X_test_copy[target_var] = list(y_test)
    #X_test_copy['ACCEPT_INDICATOR'] = list(y_test)
    X_test_copy['ACCEPT_PREDICTED_CONFIDENCE'] = model.predict_proba(X_test)[:,1]

    X_test_copy = X_test_copy.sort_values(by='ACCEPT_PREDICTED_CONFIDENCE', axis=0, ascending=False)
    # avg_acceptance_rate = y_test[y_test==1].shape[0]/y_test.shape[0]

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


def disp_timedelta(c):
    minutes, seconds = divmod(c.seconds, 60)
    hours, minutes = divmod(minutes, 60)

    # Round the microseconds to millis.
    millis = round(c.microseconds/1000, 0)

    print(f"Time taken {hours}:{minutes:02}:{seconds:02}.{millis:03}")


def generate_op(X_test, y_test, y_pred):
    cnf_matrix = metrics.confusion_matrix(y_test, y_pred)

    class_names = [0, 1]  # name  of classes
    fig, ax = plt.subplots()
    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names)
    plt.yticks(tick_marks, class_names)
    # create heatmap

    # Accuracy, Precision, Recall
    print("Accuracy:", metrics.accuracy_score(y_test, y_pred))
    print("Precision:", metrics.precision_score(y_test, y_pred))
    print("Recall:", metrics.recall_score(y_test, y_pred))
    print("F1 Score:", metrics.f1_score(y_test, y_pred))

    sns.heatmap(pd.DataFrame(cnf_matrix), annot=True, cmap="YlGnBu", fmt='g')
    ax.xaxis.set_label_position("top")
    plt.tight_layout()
    plt.title('Confusion matrix', y=1.1)
    plt.ylabel('Actual label')
    plt.xlabel('Predicted label')
    plt.show()

    # calculate roc curves
    ns_probs = [0 for _ in range(len(y_test))]
    lr_probs = model_run.predict_proba(X_test)
    ns_auc = roc_auc_score(y_test, ns_probs)
    lr_probs = lr_probs[:, 1]
    lr_auc = roc_auc_score(y_test, lr_probs)

    ns_fpr, ns_tpr, _ = roc_curve(y_test, ns_probs)
    lr_fpr, lr_tpr, _ = roc_curve(y_test, lr_probs)
    # plot the roc curve for the model
    plt.plot(ns_fpr, ns_tpr, linestyle='--', label='Random')
    plt.plot(lr_fpr, lr_tpr, marker='.', label='Model')
    # axis labels
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    # show the legend
    plt.legend()
    # show the plot
    plt.show()



    lr_precision, lr_recall, _ = precision_recall_curve(y_test, lr_probs)
    lr_f1, lr_auc = f1_score(y_test, y_pred), auc(lr_recall, lr_precision)
    # summarize scores
    #print('Model: f1=%.3f auc=%.3f' % (lr_f1, lr_auc))
    # plot the precision-recall curves
    no_skill = len(y_test[y_test==1]) / len(y_test)
    plt.plot([0, 1], [no_skill, no_skill], linestyle='--', label='Random')
    plt.plot(lr_recall, lr_precision, marker='.', label='Model')
    # axis labels
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    # show the legend
    plt.legend()
    # show the plot
    plt.show()

if __name__ == '__main__':

    # Read the Config parameters


    print(model_config)
    print(file_nm)

    df_preproc = pd.read_csv(file_nm)

    scores = []

    for i in range(9):
        print('Running for Cluster: ', i + 1)
        print('*' * 20)
        df = df_preproc[df_preproc['OFFER_ID'] == i + 1].copy()
        #target = df['ACCEPT_INDICATOR']
        target = df[target_var]
        df_trn = df.loc[:, ~df.columns.isin(['CUSTOMER_ID', 'ACCEPT_INDICATOR'])]

        X_train, X_test, y_train, y_test = train_test_split(df_trn, target, test_size=0.2, random_state=42)
        scaler = MinMaxScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)
        for model in model_config:
            # if model.get('skip','N')=='Y':
            #    continue
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

            print('Running a', model.get('model'), ' Model :\n\n')
            start_time = datetime.datetime.now()
            # parameters = {'max_depth':[d for d in range(10,110,10)], 'min_impurity_decrease':[1, 0.1, 0.01, 0.001, 0.0001, 0.00001, 0.000001]}
            # decision_tree = DecisionTreeClassifier()
            # grid = GridSearchCV(decision_tree, parameters, cv=5, scoring='roc_auc')
            parameters = model.get('parameters')
            grid = GridSearchCV(model_run, parameters, cv=5, scoring='roc_auc')
            grid.fit(X_train, y_train)

            # decision_tree = grid.best_estimator_
            # print(decision_tree)
            model_run = grid.best_estimator_
            print(model_run)
            end_time = datetime.datetime.now()
            # print("Time taken: ",str(datetime.timedelta(seconds=(end_time - start_time).seconds)))
            disp_timedelta(end_time - start_time)
            # decision_tree_score = roc_auc_score(y_test, decision_tree.predict_proba(X_test)[:,1])
            # print('ROC_AUC score for Decision Tree:', decision_tree_score)
            # scores.append(decision_tree_score)
            #
            # a, b = get_lift_gain(X_test, y_test, decision_tree)

            model_score = roc_auc_score(y_test, model_run.predict_proba(X_test)[:, 1])

            #print(model_run.pvalues)
            print('ROC_AUC score:', model_score)
            scores.append(model_score)

            a, b = get_lift_gain(X_test, y_test, model_run)

            # print(a)

            # Begin Plot Confusion Matrix
            y_pred = model_run.predict(X_test)

            y_pred_train = model_run.predict(X_train)

            print("Output on Training Data")
            generate_op(X_train, y_train, y_pred_train)

            print("Output on Test Data")
            generate_op(X_test, y_test, y_pred)



