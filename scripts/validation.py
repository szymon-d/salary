import pickle
from sklearn.metrics import recall_score, precision_score
import pathlib
import time
from sklearn.model_selection import cross_val_score
import pandas as pd


class Validation:

    def __init__(self, model_dir, dataset_dir):
        # paths
        self.model_dir = model_dir
        self.dataset_dir = dataset_dir

        # dataset
        self.dataset = {'train': [], 'test': []}

        # model
        self.model = None

        # model_parameters
        self.metrics = {'train': {'accuracy': 0, 'recall': 0, 'precision': 0},
                        'test': {'accuracy': 0, 'recall': 0, 'precision': 0}}

    def model_load(self):
        with open(self.model_dir / 'model.pkl', 'rb') as file:
            self.model = pickle.load(file)

    def dataset_load(self):
        with open(self.dataset_dir / 'X_train.pkl', 'rb') as file:
            self.dataset['train'].insert(0, pickle.load(file))
        with open(self.dataset_dir / 'X_test.pkl', 'rb') as file:
            self.dataset['test'].insert(0, pickle.load(file))
        with open(self.dataset_dir / 'y_train.pkl', 'rb') as file:
            self.dataset['train'].insert(1, pickle.load(file))
        with open(self.dataset_dir / 'y_test.pkl', 'rb') as file:
            self.dataset['test'].insert(1, pickle.load(file))

    def dataset_validation(self, y, label):
        unique_values = {label: 0 for label in y.unique()}
        print('_________________________________________________________________')
        print('{0} set contain {1} unique labels'.format(label, len(unique_values)))

        for value in unique_values:
            unique_values[value] = y[y == value].count()

        for value in unique_values.keys():
            print('{0} - {1} record'.format(value, unique_values[value]))
        print('_________________________________________________________________')

    def model_validation(self):

        for data in self.dataset.keys():
            y_pred = self.model.predict(self.dataset[data][0])
            y_true = self.dataset[data][1]

            accuracy = self.k_fold_validation(X=self.dataset[data][0], y=self.dataset[data][1])
            precision = round(precision_score(y_true, y_pred) * 100, ndigits=2)
            recall = round(recall_score(y_true, y_pred) * 100, ndigits=2)

            self.metrics[data]['accuracy'] = accuracy
            self.metrics[data]['precision'] = precision
            self.metrics[data]['recall'] = recall
            print('_________________________________________________________________')
            print('For {0} data, accuracy = {1}%, precall = {2}%, recall = {3}%'.format(data, accuracy,
                                                                                        precision, recall))
            print('_________________________________________________________________')

    def k_fold_validation(self, X, y):
        accuracy = cross_val_score(estimator=self.model, X=X, y=y, cv=10, scoring='accuracy', n_jobs=-1)
        accuracy = round(accuracy.mean() * 100, ndigits=2)
        return accuracy

    def start_validation(self):
        start = time.time()
        self.model_load()
        self.dataset_load()
        self.dataset_validation(y=pd.Series(self.dataset['train'][1]), label='Train')
        self.dataset_validation(y=pd.Series(self.dataset['test'][1]), label='Test')
        self.model_validation()
        stop = time.time()
        total_time = stop - start
        print('Validation process took {0} s'.format(round(total_time, ndigits=3)))


