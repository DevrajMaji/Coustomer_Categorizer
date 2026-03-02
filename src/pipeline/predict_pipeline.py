import os
import numpy as np

from src.utils import load_object


def load_artifacts():
    model_path = os.path.join(os.getcwd(), "artifacts", "model", "model.pkl")
    preprocessor_path = os.path.join(
        os.getcwd(), "artifacts", "data_transformation", "preprocessor.pkl"
    )
    model = load_object(model_path)
    preprocessor = load_object(preprocessor_path)
    return model, preprocessor


def predict(input_array: np.ndarray):
    model, preprocessor = load_artifacts()
    transformed = preprocessor.transform(input_array)
    return model.predict(transformed)
