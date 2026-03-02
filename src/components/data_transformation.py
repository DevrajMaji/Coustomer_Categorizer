import os
import sys
import pandas as pd
import numpy as np

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object


class DataTransformation:
    def __init__(self):
        self.transformation_dir = os.path.join(os.getcwd(),  artifacts, data_transformation)
        os.makedirs(self.transformation_dir, exist_ok=True)
        self.preprocessor_obj_path = os.path.join(self.transformation_dir, preprocessor.pkl)

    def get_data_transformer_object(self, numeric_features, categorical_features):
        try:
            num_pipeline = Pipeline([
                (imputer, SimpleImputer(strategy=median)),
                (scaler, StandardScaler())
            ])
            cat_pipeline = Pipeline([
                (imputer, SimpleImputer(strategy=most_frequent)),
                (onehot, OneHotEncoder(handle_unknown=ignore, sparse=False)),
                (scaler, StandardScaler(with_mean=False))
            ])

            preprocessor = ColumnTransformer([
                (num, num_pipeline, numeric_features),
                (cat, cat_pipeline, categorical_features)
            ])
            return preprocessor
        except Exception as e:
            raise CustomException(e, sys)

    def initiate_data_transformation(self, train_path: str, test_path: str):
        logging.info("Starting data transformation")
        try:
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            target_column = train_df.columns[-1]

            X_train = train_df.drop(columns=[target_column])
            y_train = train_df[target_column]
            X_test = test_df.drop(columns=[target_column])
            y_test = test_df[target_column]

            numeric_features = X_train.select_dtypes(include=[np.number]).columns.tolist()
            categorical_features = X_train.select_dtypes(include=[object]).columns.tolist()

            preprocessor = self.get_data_transformer_object(numeric_features, categorical_features)

            X_train_arr = preprocessor.fit_transform(X_train)
            X_test_arr = preprocessor.transform(X_test)

            save_object(self.preprocessor_obj_path, preprocessor)

            train_arr = np.c_[X_train_arr, np.array(y_train)]
            test_arr = np.c_[X_test_arr, np.array(y_test)]

            logging.info("Data transformation completed")
            return train_arr, test_arr, self.preprocessor_obj_path
        except Exception as e:
            logging.error("Error in data transformation", exc_info=True)
            raise CustomException(e, sys)
