from src.components.data_ingestion import DataIngestion
from src.components.data_transformation import DataTransformation
from src.components.model_trainer import ModelTrainer


def run_training_pipeline(raw_data_path: str):
    """Execute the full training pipeline from ingestion through model persistence."""
    # data ingestion
    ingestion = DataIngestion(file_path=raw_data_path)
    train_path, test_path = ingestion.initiate_data_ingestion()

    # data transformation
    transformation = DataTransformation()
    train_arr, test_arr, _ = transformation.initiate_data_transformation(
        train_path, test_path
    )

    # model training
    # hyperparameters chosen earlier in notebook
    model_config = {
        "C": 1000,
        "max_iter": 113,
        "multi_class": "auto",
        "penalty": "l2",
        "solver": "lbfgs",
    }
    trainer = ModelTrainer(model_config=model_config)
    metrics = trainer.initiate_model_training(train_arr, test_arr)

    print("Training metrics:", metrics)
    return metrics


if __name__ == "__main__":
    # example usage - adjust path as needed
    dataset_file = os.path.join(os.getcwd(), "data", "clustered_data.csv")
    run_training_pipeline(dataset_file)
