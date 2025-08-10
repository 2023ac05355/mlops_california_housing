import pandas as pd
import mlflow.sklearn
import pickle
from fastapi import FastAPI
from mlflow.tracking import MlflowClient
from mlflow import artifacts
import mlflow

app = FastAPI()

mlflow.set_tracking_uri("http://host.docker.internal:5000")

client = MlflowClient()

model_name = "California_Housing_Best_Model"

# Load model from registry
model = mlflow.sklearn.load_model(f"models:/{model_name}/latest")

# Get model version info for production stage
model_version_info = client.get_model_version_by_alias(model_name, "production")
run_id = model_version_info.run_id

# Download scaler and feature columns from run artifacts
scaler_path = artifacts.download_artifacts(f"runs:/{run_id}/scaler.pkl")
feature_columns_path = artifacts.download_artifacts(f"runs:/{run_id}/feature_columns.pkl")

scaler = pickle.load(open(scaler_path, "rb"))
feature_columns = pickle.load(open(feature_columns_path, "rb"))


@app.post("/predict")
def predict(input_data: dict):
    df = pd.DataFrame([input_data])
    df = pd.get_dummies(df)
    df = df.reindex(columns=feature_columns, fill_value=0)
    X_scaled = scaler.transform(df)
    prediction = model.predict(X_scaled)
    return {"prediction": prediction.tolist()}
