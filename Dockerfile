FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .
ENV MLFLOW_TRACKING_URI=http://host.docker.internal:5000
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8080"]
