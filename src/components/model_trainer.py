import os
import sys
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object


class ModelTrainer:
    """Train a classification model and save the artifact."""

    def __init__(self, model_config: dict = None):
        self.model_dir = os.path.join(os.getcwd(), "artifacts", "model")
        os.makedirs(self.model_dir, exist_ok=True)
        self.model_path = os.path.join(self.model_dir, "model.pkl")
        self.model_config = model_config or {}

    def initiate_model_training(self, train_array, test_array):
        try:
            X_train = train_array[:, :-1]
            y_train = train_array[:, -1]
            X_test = test_array[:, :-1]
            y_test = test_array[:, -1]

            # instantiate the logistic regression model with provided hyperparameters
            model = LogisticRegression(**self.model_config)
            model.fit(X_train, y_train)

            preds = model.predict(X_test)
            accuracy = accuracy_score(y_test, preds)
            f1 = f1_score(y_test, preds, average="weighted")

            logging.info(
                f"Model training completed. Accuracy: {accuracy:.4f}, F1: {f1:.4f}"
            )

            save_object(self.model_path, model)

            return {"accuracy": accuracy, "f1": f1, "model_path": self.model_path}
        except Exception as e:
            logging.error("Error during model training", exc_info=True)
            raise CustomException(e, sys)
