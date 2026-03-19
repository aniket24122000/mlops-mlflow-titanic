import mlflow.sklearn
import pandas as pd
import os
import joblib
import mlflow
from dotenv import load_dotenv

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

from src.logger import get_logger

# load env
load_dotenv()

logger = get_logger()

# MLflow settings
mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI"))
mlflow.set_experiment(os.getenv("MLFLOW_EXPERIMENT_NAME"))

# authentication
os.environ["MLFLOW_TRACKING_USERNAME"] = os.getenv("MLFLOW_TRACKING_USERNAME")
os.environ["MLFLOW_TRACKING_PASSWORD"] = os.getenv("MLFLOW_TRACKING_PASSWORD")


def train_model(input_path):

    logger.info("Starting Model Training")

    df = pd.read_csv(input_path)

    X = df.drop("survived", axis=1)
    y = df["survived"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    with mlflow.start_run():

        model = RandomForestClassifier()

        model.fit(X_train, y_train)

        preds = model.predict(X_test)

        acc = accuracy_score(y_test, preds)

        logger.info(f"Accuracy: {acc}")

        # log parameter
        mlflow.log_param("model", "RandomForest")

        # log metric
        mlflow.log_metric("accuracy", acc)

        # save local artifact
        os.makedirs("artifacts", exist_ok=True)
        model_path = "artifacts/model.pkl"
        joblib.dump(model, model_path)

        # log artifact (optional)
        mlflow.log_artifact(model_path)

        #log model
        mlflow.sklearn.log_model(
            sk_model=model,
            artifact_path="model",
            registered_model_name="titanic_model"
        )

        logger.info("Model logged and registered successfully")

    return acc