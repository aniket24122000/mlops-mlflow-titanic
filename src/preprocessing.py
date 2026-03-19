import pandas as pd
import os
from src.logger import get_logger

logger = get_logger()

def run_preprocessing(input_path):

    logger.info("Starting preprocessing")

    df = pd.read_csv(input_path)

    df = df[["survived", "sex", "pclass", "age", "fare", "embarked"]]

    df.dropna(inplace=True)

    os.makedirs("data/processed", exist_ok=True)

    output_path = "data/processed/preprocessed.csv"

    df.to_csv(output_path, index=False)

    logger.info(f"Preprocessed data saved at {output_path}")

    return output_path