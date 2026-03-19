# Titanic Survival Prediction API (MLOps Project)
---

# Project Overview

This project demonstrates an **end-to-end MLOps pipeline** for predicting whether a passenger survived the Titanic disaster.

The system includes:

* Data ingestion pipeline
* Data preprocessing
* Feature engineering
* Model training
* Experiment tracking using **MLflow**
* Model registry using **DagsHub**
* Prediction API using **FastAPI**
* Containerization using **Docker**
* Automated **CI/CD pipeline using GitHub Actions**

---

# Architecture

```
                +------------------+
                |   Raw Dataset     |
                +------------------+
                         |
                         v
              +---------------------+
              |  Data Preprocessing |
              +---------------------+
                         |
                         v
              +---------------------+
              |  Feature Engineering|
              +---------------------+
                         |
                         v
                +------------------+
                |  Model Training  |
                +------------------+
                         |
                         v
                +------------------+
                | MLflow Tracking  |
                +------------------+
                         |
                         v
                +------------------+
                |  Model Artifact  |
                +------------------+
                         |
                         v
                +------------------+
                |   FastAPI API    |
                +------------------+
                         |
                         v
                +------------------+
                |  Docker Container|
                +------------------+
                         |
                         v
                +------------------+
                |  Deployment API  |
                +------------------+
```

---

# Project Structure

```
mlops-project
│
├── app
│   ├── main.py
│   ├── schema.py
│   └── __init__.py
│
├── src
│   ├── data_ingestion.py
│   ├── preprocessing.py
│   ├── feature_engineering.py
│   ├── train.py
│   └── logger.py
│
├── tests
│   └── test_api.py
│
├── artifacts
│   ├── model.pkl
│   └── logs
│
├── data
│
├── Dockerfile
├── requirements-api.txt
├── requirements.txt
├── run_pipeline.py
├── .env
└── README.md
```

---

# MLflow Experiment Tracking

Experiments are tracked using **MLflow integrated with DagsHub**.

Tracked information includes:

* Model parameters
* Evaluation metrics
* Model artifacts
* Training runs

Example MLflow experiment dashboard:

![MLflow Experiment](mlflow_experiment.png)

---

# Setup Instructions

## Clone Repository

```
git clone <your-repository-url>
cd mlops-project
```

---

# Install Dependencies

```
pip install -r requirements.txt
```

API dependencies:

```
pip install -r requirements-api.txt
```

---

# Run Training Pipeline

```
python run_pipeline.py
```

Pipeline steps:

* data ingestion
* preprocessing
* feature engineering
* model training
* MLflow experiment logging

Model will be saved in:

```
artifacts/model.pkl
```

---

# Run FastAPI Server

```
uvicorn app.main:app --reload
```

Server will run at:

```
http://localhost:8000
```

---

# Test API via Swagger UI

Open in browser:

```
http://localhost:8000/docs
```

Example request:

```
{
 "sex": 1,
 "pclass": 3,
 "age": 25,
 "fare": 7.25,
 "embarked": 0
}
```

Example response:

```
{
 "prediction": 1
}
```

---

# Run Tests

```
pytest
```

---

# Docker Deployment

## Build Docker Image

```
docker build -t titanic-api .
```

---

## Run Docker Container

```
docker run -p 8000:8000 titanic-api
```

Access API:

```
http://localhost:8000/docs
```

---

# CI/CD Pipeline

This project includes **GitHub Actions CI/CD pipeline**.

Pipeline automatically:

1. installs dependencies
2. runs tests
3. builds Docker image
4. pushes image to DockerHub

Workflow location:

```
.github/workflows/ci-cd.yml
```

---

# Technologies Used

* Python
* FastAPI
* MLflow
* DagsHub
* Scikit-learn
* Docker
* GitHub Actions
* Pytest

---

#Virtual Server Image 
<img width="1868" height="992" alt="image" src="https://github.com/user-attachments/assets/4d6046c7-e5f3-434a-ab86-1d2068160028" />

<img width="1907" height="941" alt="image" src="https://github.com/user-attachments/assets/a53b0b52-bec4-409c-a0fc-a5de09388bf6" />

#MLFLOW Dagshub 
DagsHUB url : https://dagshub.com/aniketkandrikar29/mlops-mlflow-titanic.mlflow/#/experiments/2/runs?searchFilter=&orderByKey=attributes.start_time&orderByAsc=false&startTime=ALL&lifecycleFilter=Active&modelVersionFilter=All+Runs&datasetsFilter=W10%3D
<img width="1918" height="980" alt="image" src="https://github.com/user-attachments/assets/eca80e45-1843-4727-b2c2-7d2978ccdbdc" />

