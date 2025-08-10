import os
import pandas as pd
import mlflow.sklearn
import pickle
from fastapi import FastAPI
from mlflow.tracking import MlflowClient
from mlflow import artifacts
import mlflow

app = FastAPI()

mlflow_tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "databricks")
mlflow.set_tracking_uri(mlflow_tracking_uri)

client = MlflowClient()

model_name = "mlops.default.california_housing_best_model"

# Load model by alias "production" using '@' syntax (this is supported)
model = mlflow.sklearn.load_model(f"models:/{model_name}@production")

# To get model version info, list all versions and filter manually (avoid get_latest_versions)
all_versions = client.search_model_versions(f"name='{model_name}'")

prod_version_info = None
for v in all_versions:
    # Fetch full model version info (includes aliases)
    full_version = client.get_model_version(name=model_name, version=v.version)
    if 'production' in full_version.aliases:
        prod_version_info = full_version
        break

if prod_version_info is None:
    raise Exception("Production alias not found for model")

run_id = prod_version_info.run_id
# Download artifacts associated with this run
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
