import spacy
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

emotions = ['love','anger','fear','surprise','joy','sadness']

df_love = pd.read_csv('C:/users/kecco/Documenti/Github/DataMining-EmotionDetection/data/Processed/Dataset/StackOverflow/love.csv',sep=';',encoding='iso-8859-1')

nlp = spacy.load("en_core_web_trf")

sentences = df_love['Text']
vectorized = []

for i,sentence in enumerate(sentences):
    print(i)
    doc = nlp(sentence)
    v = doc.vector
    vectorized.append(v)

x = vectorized
y = df_love['label']
y = y.fillna(0)
y = y.map({emotions[1].upper(): 1, 0: 0}).astype(int)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=1, stratify=y)
#%%
from imblearn.over_sampling import SMOTE

sm = SMOTE(sampling_strategy=1, random_state=10)
x_train = np.array(x_train)
x_smote, y_smote = sm.fit_resample(x_train, y_train)

from sklearn.svm import LinearSVC

from sklearn.model_selection import GridSearchCV

svm = LinearSVC(max_iter=10000)  # class_weight='balanced'
scoring = ['f1', 'roc_auc']
#param_grid = {'C': [0.01, 0.05, 0.10, 0.20, 0.50, 1, 2, 4, 8]}
param_grid = {'C': [0.05,1,8]}

gs = GridSearchCV(svm, param_grid, scoring=scoring,
                      refit='roc_auc', cv=10, verbose=2, n_jobs=-1,
                      return_train_score=True)

gs.fit(x_smote, y_smote)

# print best parameter after tuning
print("\nBest parameter GridSearchCV: ")
print(gs.best_params_)

from sklearn.metrics import classification_report
from sklearn.metrics import roc_auc_score, f1_score
import scikitplot as skplt
import matplotlib.pyplot as plt

grid_predictions = gs.predict(x_test)

# print classification report
print(classification_report(y_test, grid_predictions))
rfc = gs.best_estimator_
print("\nC parameter's value used: ")
print(rfc)
print("\nAUC: ")
print(roc_auc_score(y_test, grid_predictions))
print("\nF1: ")
print(f1_score(y_test, grid_predictions))
skplt.metrics.plot_confusion_matrix(grid_predictions, y_test)
plt.show()
