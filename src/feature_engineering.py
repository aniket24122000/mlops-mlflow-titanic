import pandas as pd
import os
from sklearn.preprocessing import LabelEncoder
from src.logger import get_logger

logger = get_logger()

def run_feature_engineering(input_path):

    logger.info("Starting feature engineering")

    df = pd.read_csv(input_path)

    encoder = LabelEncoder()

    df["sex"] = encoder.fit_transform(df["sex"])
    df["embarked"] = encoder.fit_transform(df["embarked"])

    os.makedirs("artifacts", exist_ok=True)

    feature_path = "data/processed/features.csv"

    df.to_csv(feature_path, index=False)

    logger.info(f"Feature dataset saved at {feature_path}")

    return feature_path