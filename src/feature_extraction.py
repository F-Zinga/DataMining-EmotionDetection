import numpy as np
import pandas as pd
from tokenizer import tokenize


def word_check(word, features):
    flag = False
    for feature in features:
        if word == feature:
            flag = True
    return flag


def load_dict(path, sp):

    diz = pd.read_csv(path, sep=sp)
    features = diz._get_column_array(0)

    return features, diz


def mix_diz(features1, features2):

    for feature in features2:
        if feature not in features1:
            features1 = np.append(features1, feature)

    return features1


def wordcount(dataframe, features, diz):

    sentences = dataframe._get_column_array(0)
    dim = len(sentences)
    df = pd.DataFrame(0.000, index=range(0, dim), columns=features)
    row = 0

    docs = tokenize(sentences)
    for doc in docs:
        for tok in doc:
            word = tok.lower_
            flag = word_check(word, features)
            if flag:
                df.at[row, word] += 1
        row += 1

    return df


# in case of using NRC Intensity Score
def update(dataframe_to_update, diz, word, row):

    app = diz[diz['word'] == word]
    score = app.iloc[0]['score']
    dataframe_to_update.at[row, word] += score

