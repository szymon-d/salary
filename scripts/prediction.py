import pathlib
import os
import pickle
import numpy as np

def model_load(model_dir):
    with open(model_dir / 'model.pkl', 'rb') as file:
        model = pickle.load(file)
        return model


def parameters_load(parameters_dir):
    with open(parameters_dir / 'scaler_x.pkl', 'rb') as file:
        sc_x = pickle.load(file)

    with open(parameters_dir / 'all_features.pkl', 'rb') as file:
        all_features = pickle.load(file)
    return sc_x, all_features


def data_check(value1, value2, all_features):
    if value1 and value2 != 'O':
        to_pred = np.array([[value1, value2]], dtype='float32')
    return to_pred


def data_preprocessing(scaler, to_pred):
    to_pred = scaler.transform(to_pred)
    return to_pred


def make_prediction(model, to_pred):
    prediction = model.predict(to_pred)
    return prediction




model = model_load(model_dir = pathlib.Path.cwd().parents[0]/'.tox'/'SVM_model'/'model')
sc_x, all_features = parameters_load(parameters_dir = pathlib.Path.cwd().parents[0]/'.tox'/'SVM_model'/'parameters')

value1 = input('Provide age: ')
value2 = input('Provide salary :')

to_pred = data_check(value1,value2, all_features)
to_pred = data_preprocessing(sc_x, to_pred)

predict = int(make_prediction(model, to_pred))

mapper = {0: 'This person doesnt cheat', 1: 'There is a risk this person can cheat'}

print(mapper[predict])
