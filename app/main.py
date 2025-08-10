import os
import pandas as pd
import mlflow.sklearn
import pickle
from fastapi import FastAPI
from mlflow.tracking import MlflowClient
from mlflow import artifacts
import mlflow
import logging
import sqlite3
from datetime import datetime

logging.basicConfig(filename='app.log', level=logging.INFO, format='%(asctime)s %(message)s')

app = FastAPI()

mlflow_tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "databricks")
mlflow.set_tracking_uri(mlflow_tracking_uri)
client = MlflowClient()
model_name = "mlops.default.california_housing_best_model"
model = mlflow.sklearn.load_model(f"models:/{model_name}@production")
all_versions = client.search_model_versions(f"name='{model_name}'")
prod_version_info = None
for v in all_versions:
    # Fetch full model version info
    full_version = client.get_model_version(name=model_name, version=v.version)
    if 'production' in full_version.aliases:
        prod_version_info = full_version
        break
if prod_version_info is None:
    raise Exception("Production alias not found for model")

run_id = prod_version_info.run_id
scaler_path = artifacts.download_artifacts(f"runs:/{run_id}/scaler.pkl")
feature_columns_path = artifacts.download_artifacts(f"runs:/{run_id}/feature_columns.pkl")
scaler = pickle.load(open(scaler_path, "rb"))
feature_columns = pickle.load(open(feature_columns_path, "rb"))

# Initialize SQLite connection and create table
conn = sqlite3.connect('logs.db', check_same_thread=False)
cursor = conn.cursor()
cursor.execute('''
CREATE TABLE IF NOT EXISTS logs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp TEXT,
    request_data TEXT,
    prediction_output TEXT
)
''')
conn.commit()


@app.post("/predict")
async def predict(input_data: dict):
    logging.info(f"Received request data: {input_data}")
    # Prepare input data
    df = pd.DataFrame([input_data])
    df = pd.get_dummies(df)
    df = df.reindex(columns=feature_columns, fill_value=0)

    X_scaled = scaler.transform(df)
    prediction = model.predict(X_scaled)

    # Log to SQLite
    cursor.execute(
        "INSERT INTO logs (timestamp, request_data, prediction_output) VALUES (?, ?, ?)",
        (datetime.utcnow().isoformat(), str(input_data), str(prediction.tolist()))
    )
    conn.commit()
    logging.info(f"Model output: {prediction}")
    return {"prediction": prediction.tolist()}
