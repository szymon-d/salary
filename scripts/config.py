import pathlib
import os

MODEL_NAME = 'SVM_model'
ROOT_DIR = pathlib.Path.cwd()/'.tox'/MODEL_NAME
DATASET_DIR = ROOT_DIR/'dataset'
PARAMETERS_DIR = ROOT_DIR/'parameters'
MODEL_DIR = ROOT_DIR/'model'

def create_directory():
    os.mkdir(DATASET_DIR)
    os.mkdir(PARAMETERS_DIR)
    os.mkdir(MODEL_DIR)
