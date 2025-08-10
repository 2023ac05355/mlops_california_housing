#!/bin/bash
set -e

IMAGE="${DOCKERHUB_USERNAME}/housing-api:latest"

echo "Using image: $IMAGE"

echo "Pulling latest image..."
docker pull "$IMAGE"

echo "Stopping old container..."
docker stop housing-api || true
docker rm housing-api || true

echo "Starting new container..."
docker run -d -p 8080:8080 \
  -e MLFLOW_TRACKING_URI="$MLFLOW_TRACKING_URI" \
  -e DATABRICKS_HOST="$DATABRICKS_HOST" \
  -e DATABRICKS_TOKEN="$DATABRICKS_TOKEN" \
  --name housing-api "$IMAGE"

echo "Done!"