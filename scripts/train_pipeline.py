import config
import pipeline
import validation

config.create_directory()

df = pipeline.data_download()
pipeline.variables(df)
X_train, y_train = pipeline.data_split(df)
pipeline.model_train(X_train, y_train)
validation_object = validation.Validation(model_dir = config.MODEL_DIR , dataset_dir = config.DATASET_DIR)
validation_object.start_validation()
