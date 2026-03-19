import mlflow.pyfunc
import pandas as pd
import os

from fastapi import FastAPI
from dotenv import load_dotenv
from fastapi.middleware.cors import CORSMiddleware

from app.schema import Passenger

load_dotenv()

app = FastAPI(title="Titanic Production Model API")

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # production me specific domain use karte hain
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# MLflow authentication
os.environ["MLFLOW_TRACKING_USERNAME"] = os.getenv("MLFLOW_TRACKING_USERNAME")
os.environ["MLFLOW_TRACKING_PASSWORD"] = os.getenv("MLFLOW_TRACKING_PASSWORD")

# tracking server
mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI"))

MODEL_NAME = os.getenv("MODEL_NAME")
MODEL_STAGE = os.getenv("MODEL_STAGE")

# load production model
model = mlflow.pyfunc.load_model(
    model_uri=f"models:/{MODEL_NAME}/{MODEL_STAGE}"
)


@app.get("/")
def home():
    return {"message": "Production Titanic Model API Running"}


@app.post("/predict")
def predict(data: Passenger):

    input_df = pd.DataFrame([{
        "sex": data.sex,
        "pclass": data.pclass,
        "age": data.age,
        "fare": data.fare,
        "embarked": data.embarked
    }])

    prediction = model.predict(input_df)

    result = int(prediction[0])

    return {
        "prediction": result,
        "result": "Survived" if result == 1 else "Not Survived"
    }