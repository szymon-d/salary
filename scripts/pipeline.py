import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import pickle
import os
import config
import pathlib
from sklearn.svm import SVC

def data_download():
    df = pd.read_csv(pathlib.Path.cwd()/'Social_Network_Ads.csv', sep = ',', header = 0)
    with open(config.DATASET_DIR/'df.pkl', 'wb') as file:
        pickle.dump(df, file)
    return df



def variables(df):
    all_features = df.columns[:-1]
    categorical_features = [var for var in all_features if df[var].dtype == 'O']
    numerical_features = [var for var in all_features if df[var].dtype != 'O']

    with open(config.PARAMETERS_DIR / 'all_features.pkl', 'wb') as file:
        pickle.dump(all_features, file)
    with open(config.PARAMETERS_DIR / 'categorical_features.pkl', 'wb') as file:
        pickle.dump(categorical_features, file)
    with open(config.PARAMETERS_DIR / 'numerical_features.pkl', 'wb') as file:
        pickle.dump(numerical_features, file)


def data_split(df):
    X = np.array(df.iloc[:,:-1]).astype('float32')
    y = np.array(df.iloc[:,-1]).astype('float32')

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

    sc_x = StandardScaler()
    sc_x.fit(X_train)
    X_test = sc_x.transform(X_test)
    X_train = sc_x.transform(X_train)

    with open(config.DATASET_DIR / 'X_train.pkl', 'wb') as file:
        pickle.dump(X_train, file)
    with open(config.DATASET_DIR / 'X_test.pkl', 'wb') as file:
        pickle.dump(X_test, file)
    with open(config.DATASET_DIR / 'y_train.pkl', 'wb') as file:
        pickle.dump(y_train, file)
    with open(config.DATASET_DIR / 'y_test.pkl', 'wb') as file:
        pickle.dump(y_test, file)
    with open(config.PARAMETERS_DIR / 'scaler_x.pkl', 'wb') as file:
        pickle.dump(sc_x, file)

    return X_train, y_train


def model_train(X_train, y_train):
    model = SVC(C=0.8, kernel='rbf', degree=4)
    model.fit(X_train, y_train)

    with open(config.MODEL_DIR/'model.pkl', 'wb') as file:
        pickle.dump(model, file)

