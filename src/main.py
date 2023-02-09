import tuning as t
import feature_extraction as fe
import sampling as s
import evaluation as e
from pathlib import Path
from sklearn.svm import LinearSVC
from sklearn.dummy import DummyClassifier
from sklearn.metrics import classification_report

# TEST EMOTION
emo = 'anger'

# folder 2, enable in case of mixed lexicon
# lex_folder1 = Path('/Lexicon/WordnetCategories/')
lex_folder = Path('/Lexicon/NRC/')
dataset_folder = Path('/Dataset/StackOverflow')

file_name = emo + '.csv'
svm = LinearSVC(max_iter=10000)  # class_weight='balanced'


# Stratified Sampling

sep = ';'  # separator
dataset = s.read_dataset(dataset_folder / file_name, sep)
x, y = s.set_xy(dataset)
emo_up = emo.upper()
y = y.fillna(0)
y = y.map({emo_up: 1, 0: 0}).astype(int)    # for scoring parameter
x_train, x_test, y_train, y_test = s.stratified_sampling(x, y)


# Feature Extraction on Train set for training the model

features, diz = fe.load_dict(lex_folder / file_name, sep)

# MIX LEXICONS
# features1, diz2 = fe.load_dict(lex_folder1 / file_name, sep)
# features = fe.mix_diz(features, features1)

x_train = fe.wordcount(x_train, features, diz)


# SMOTE

x_train, y_train = s.smote(x_train, y_train)


# Parameter tuning

model, train_score = t.hypertuning(svm, x_train, y_train)


# Feature Extraction on Test set for evaluation

x_test = fe.wordcount(x_test, features, diz)

# Evaluation

test_score = e.evaluate(model, x_test, y_test)
print("\nSCORE:")
print(test_score)


# Baseline

dummy_classifier = DummyClassifier()
dummy_classifier.fit(x_train, y_train)
dummy_predictions = dummy_classifier.predict(x_test)
print(classification_report(y_test, dummy_predictions))
