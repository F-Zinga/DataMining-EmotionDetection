import pandas as pd
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score


def read_csv():

    emotions_list = ['love','anger','fear','surprise','joy','sadness']

    df_love = pd.read_csv('C:/users/kecco/Documenti/Github/DataMining-EmotionDetection/data/Processed/Dataset/StackOverflow/love.csv',sep=';',encoding='iso-8859-1')

    df_anger = pd.read_csv('C:/users/kecco/Documenti/Github/DataMining-EmotionDetection/data/Processed/Dataset/StackOverflow/anger.csv',sep=';',encoding='iso-8859-1')

    df_fear = pd.read_csv('C:/users/kecco/Documenti/Github/DataMining-EmotionDetection/data/Processed/Dataset/StackOverflow/fear.csv',sep=';',encoding='iso-8859-1')

    df_surprise = pd.read_csv('C:/users/kecco/Documenti/Github/DataMining-EmotionDetection/data/Processed/Dataset/StackOverflow/surprise.csv',sep=';',encoding='iso-8859-1')

    df_joy = pd.read_csv('C:/users/kecco/Documenti/Github/DataMining-EmotionDetection/data/Processed/Dataset/StackOverflow/joy.csv',sep=';',encoding='iso-8859-1')

    df_sadness = pd.read_csv('C:/users/kecco/Documenti/Github/DataMining-EmotionDetection/data/Processed/Dataset/StackOverflow/sadness.csv',sep=';',encoding='iso-8859-1')

    df_list = [df_love, df_anger, df_fear, df_surprise, df_joy,df_sadness]

    return df_list, emotions_list


def set_labels(df,i,emotions_list):

    y = df['label']
    y = y.fillna(0)
    y = y.map({emotions_list[i].upper(): 1, 0: 0}).astype(int)

    return y


def classification_report_with_accuracy_score(y_test, y_pred):

    print(classification_report(y_test, y_pred)) # print classification report
    return accuracy_score(y_test, y_pred) # return accuracy score