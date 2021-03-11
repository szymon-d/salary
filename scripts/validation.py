from sklearn.metrics import accuracy_score, precision_score, recall_score
import pickle
import config

def data_gathering():
    with open('train_test_data/X_train.pkl', 'rb') as file:
        X_train = pickle.load(file)
    with open('train_test_data/X_test.pkl', 'rb') as file:
        X_test = pickle.load(file)
    with open('train_test_data/y_train.pkl', 'rb') as file:
        y_train = pickle.load(file)
    with open('train_test_data/y_test.pkl', 'rb') as file:
        y_test = pickle.load(file)

    return X_train, X_test, y_train, y_test


def model_validation():
    X_train, X_test, y_train, y_test = data_gathering()
    dataset = {'train': [X_train, y_train], 'test': [X_test, y_test]}

    with open(config.MODEL_DIR/'model.pkl', 'rb') as file:
        model = pickle.load(file)

    for data in dataset.keys():
        y_pred = model.predict(dataset[data][0])
        y_true = dataset[data][1]

        accuracy = round(accuracy_score(y_true, y_pred) * 100, ndigits=2)
        precision = round(precision_score(y_true, y_pred) * 100, ndigits=2)
        recall = round(recall_score(y_true, y_pred) * 100, ndigits=2)

        print('For {0} data, accuracy = {1}%, precall = {2}%, recall = {3}'.format(data, accuracy, precision, recall))