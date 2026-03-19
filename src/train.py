import pandas as pd
import os
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from src.logger import get_logger

logger = get_logger()

def train_model(input_path):

    logger.info("Starting Model Training")

    df = pd.read_csv(input_path)

    X = df.drop("survived", axis=1)
    y = df["survived"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = RandomForestClassifier()

    model.fit(X_train, y_train)

    preds = model.predict(X_test)

    acc = accuracy_score(y_test, preds)

    logger.info(f"Model Accuracy: {acc}")

    os.makedirs("artifacts", exist_ok=True)

    model_path = "artifacts/model.pkl"

    joblib.dump(model, model_path)

    logger.info(f"Model saved at {model_path}")

    return acc