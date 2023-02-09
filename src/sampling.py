import numpy as np
from sklearn.model_selection import train_test_split
import pandas as pd
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler


def read_dataset(path, sp):
    dataset = pd.read_csv(path, sep=sp)
    return dataset


def set_xy(df):
    y = df['label']
    x = df.drop(['label'], axis=1)
    return x, y


def stratified_sampling(x, y):
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=1, stratify=y)
    return x_train, x_test, y_train, y_test


def random_us(x, y):
    rus = RandomUnderSampler(random_state=10)
    x = np.array(x)
    x_res, y_res = rus.fit_resample(x, y)
    return x_res, y_res


def smote(x, y):
    sm = SMOTE(sampling_strategy=1, random_state=10)
    x = np.array(x)
    x_smote, y_smote = sm.fit_resample(x, y)

    return x_smote, y_smote
