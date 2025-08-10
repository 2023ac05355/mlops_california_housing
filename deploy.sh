#!/bin/bash
set -e

IMAGE="${DOCKERHUB_USERNAME}/housing-api:latest"

echo "Pulling latest image..."
docker pull $IMAGE

echo "Stopping old container..."
docker stop housing-api || true
docker rm housing-api || true

echo "Starting new container..."
docker run -d -p 8080:8080 --name housing-api $IMAGE
