import pandas as pd
import mlflow.sklearn
import pickle
from fastapi import FastAPI, Request, BackgroundTasks
from mlflow.tracking import MlflowClient
from mlflow import artifacts
import mlflow
import logging
import sqlite3
from datetime import datetime
import time
from pydantic import BaseModel, Field
from prometheus_client import Counter, Histogram
from typing import Literal

logging.basicConfig(filename='app.log', level=logging.INFO, format='%(asctime)s %(message)s')

# Prometheus metrics
REQUEST_COUNT = Counter("request_count", "Total number of requests")
REQUEST_LATENCY = Histogram("request_latency_seconds", "Latency of requests in seconds")

app = FastAPI()


class HousingInput(BaseModel):
    longitude: float = Field(..., ge=-125, le=-114)
    latitude: float = Field(..., ge=32, le=42)
    housing_median_age: float = Field(..., gt=0)
    total_rooms: float = Field(..., gt=0)
    total_bedrooms: float = Field(..., gt=0)
    population: float = Field(..., gt=0)
    households: float = Field(..., gt=0)
    median_income: float = Field(..., gt=0)
    ocean_proximity: Literal[
        "NEAR BAY",
        "INLAND",
        "<1H OCEAN",
        "NEAR OCEAN",
        "ISLAND"
    ]


mlflow_tracking_uri = "databricks"
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


@REQUEST_LATENCY.time()
@app.post("/predict")
async def predict(input_data: HousingInput):
    logging.info(f"Received request data: {input_data}")
    REQUEST_COUNT.inc()
    # Convert pydantic model to dict
    data_dict = input_data.dict()
    df = pd.DataFrame([data_dict])
    df = pd.get_dummies(df)
    df = df.reindex(columns=feature_columns, fill_value=0)
    X_scaled = scaler.transform(df)
    # Predict
    prediction = model.predict(X_scaled)

    # Log to SQLite
    cursor.execute(
        "INSERT INTO logs (timestamp, request_data, prediction_output) VALUES (?, ?, ?)",
        (datetime.utcnow().isoformat(), str(data_dict), str(prediction.tolist()))
    )
    conn.commit()

    logging.info(f"Model output: {prediction.tolist()}")
    return {"prediction": prediction.tolist()}


request_count = 0
total_latency = 0.0


@app.middleware("http")
async def metrics_middleware(request: Request, call_next):
    global request_count, total_latency
    start_time = time.time()
    response = await call_next(request)
    process_time = time.time() - start_time

    request_count += 1
    total_latency += process_time

    return response


@app.get("/metrics")
def get_metrics():
    avg_latency = total_latency / request_count if request_count > 0 else 0
    return {
        "total_requests": request_count,
        "average_latency_seconds": round(avg_latency, 4)
    }


def retrain_model():
    print("Starting model retraining...")
    time.sleep(5) 
    print("Model retraining completed!")


@app.post("/new-data")
def new_data_endpoint(data: HousingInput, background_tasks: BackgroundTasks):
    background_tasks.add_task(retrain_model)
    return {"message": "New data received. Model retraining triggered in background."}
