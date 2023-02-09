from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_score
import pandas as pd
import numpy as np
from sklearn import metrics


def hypertuning(model, x_train, y_train):

    scoring = ['f1', 'roc_auc']
    param_grid = {'C': [0.01, 0.05, 0.10, 0.20, 0.50, 1, 2, 4, 8]}

    gs = GridSearchCV(model, param_grid, scoring=scoring,
                      refit='roc_auc', cv=10, verbose=2, n_jobs=-1,
                      return_train_score=True)

    gs.fit(x_train, y_train)

    # print best parameter after tuning
    print("\nBest parameter GridSearchCV: ")
    print(gs.best_params_)

    return gs, gs.score(x_train, y_train)


# A function that compares the CV performance of a set of predetermined models
def cv_comparison(models, x, y, cv):
    # Initiate a DataFrame for the averages and a list for all measures
    cv_accuracies = pd.DataFrame()
    precisions = []
    recalls = []
    f1s = []
    # Loop through the models, run a CV, add the average scores to the DataFrame and the scores of
    # all CVs to the list
    for model in models:
        precision = -np.round(cross_val_score(model, x, y, scoring='precision', cv=cv), 4)
        precisions.append(precision)
        precision_avg = round(precision.mean(), 4)
        recall = -np.round(cross_val_score(model, x, y, scoring='recall', cv=cv), 4)
        recalls.append(recall)
        recall_avg = round(recall.mean(), 4)
        f1 = np.round(cross_val_score(model, x, y, scoring='f1', cv=cv), 4)
        f1s.append(f1)
        f1_avg = round(f1.mean(), 4)
        cv_accuracies[str(model)] = [precision_avg, recall_avg, f1_avg]
    cv_accuracies.index = ['Precision', 'Recall', 'F1 Score']
    return cv_accuracies, precisions, recalls, f1s
