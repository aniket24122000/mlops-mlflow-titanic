import pandas as pd
import os
from sklearn.datasets import fetch_openml
from src.logger import get_logger

logger = get_logger()

def run_data_ingestion():

    logger.info("Starting Data Ingestion")

    os.makedirs("data/raw", exist_ok=True)

    data = fetch_openml(name="titanic", version=1, as_frame=True)

    df = data.frame

    df.to_csv("data/raw/titanic_raw.csv", index=False)

    logger.info("Raw dataset saved to data/raw/titanic_raw.csv")

    return "data/raw/titanic_raw.csv"


if __name__ == "__main__":
    run_data_ingestion()