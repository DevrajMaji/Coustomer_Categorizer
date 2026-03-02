# Customer Categorizer

This project demonstrates a machine learning workflow for customer categorization. It includes data ingestion, transformation, model training, and a simple Flask deployment.

## Setup

1. Create and activate a Python environment (venv, conda, etc.).
2. Install dependencies:
   `sh
   pip install -r requirements.txt
   `

## Training

Run the training pipeline which will ingest raw data, transform it, train a logistic regression model, and save artifacts:

`sh
python -m src.pipeline.train_pipeline
`

The trained model will be saved to rtifacts/model/model.pkl.

## Serving the Model

Start the Flask app (after training):

`sh
python app.py
`

- GET / – health check
- POST /predict – send JSON of features (list or dict) to obtain prediction

Example payload:

`json
[feature1, feature2, ...]
`

or

`json
{ feature_a: value, feature_b: value, ...}
`

'
