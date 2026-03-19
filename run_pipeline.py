from src.data_ingestion import run_data_ingestion
from src.preprocessing import run_preprocessing
from src.feature_engineering import run_feature_engineering
from src.train import train_model

def run_pipeline():

    raw_data = run_data_ingestion()

    processed_data = run_preprocessing(raw_data)

    features = run_feature_engineering(processed_data)

    accuracy = train_model(features)

    print("Pipeline completed")
    print("Model accuracy:", accuracy)


if __name__ == "__main__":
    run_pipeline()