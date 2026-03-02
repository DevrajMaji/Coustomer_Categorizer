import os
import sys
import pandas as pd
from sklearn.model_selection import train_test_split

from src.exception import CustomException
from src.logger import logging


class DataIngestion:
    """Handles reading raw data, splitting into train/test sets, and saving artifacts."""

    def __init__(self, file_path: str, test_size: float = 0.2, random_state: int = 42):
        self.file_path = file_path
        self.test_size = test_size
        self.random_state = random_state

        # prepare directories for saving ingested data
        self.ingestion_dir = os.path.join(os.getcwd(), "artifacts", "data_ingestion")
        os.makedirs(self.ingestion_dir, exist_ok=True)

        self.raw_data_path = os.path.join(self.ingestion_dir, "raw.csv")
        self.train_data_path = os.path.join(self.ingestion_dir, "train.csv")
        self.test_data_path = os.path.join(self.ingestion_dir, "test.csv")

    def initiate_data_ingestion(self):
        logging.info("Starting data ingestion process")
        try:
            # read source file
            df = pd.read_csv(self.file_path)
            logging.info(f"Loaded raw data from {self.file_path} with shape {df.shape}")

            # save a copy of the raw data
            df.to_csv(self.raw_data_path, index=False)
            logging.info(f"Saved raw data to {self.raw_data_path}")

            # split into train and test
            train_set, test_set = train_test_split(
                df, test_size=self.test_size, random_state=self.random_state
            )
            train_set.to_csv(self.train_data_path, index=False)
            test_set.to_csv(self.test_data_path, index=False)

            logging.info(
                f"Data ingestion completed. Train shape: {train_set.shape}, test shape: {test_set.shape}"
            )

            return self.train_data_path, self.test_data_path

        except Exception as e:
            logging.error("Error during data ingestion", exc_info=True)
            raise CustomException(e, sys)