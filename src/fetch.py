import os
import kagglehub
from pandas import DataFrame, pandas as pd

DATASET = "laotse/credit-risk-dataset"

def fetch_dataset() -> DataFrame:
    dataset_dir = kagglehub.dataset_download("laotse/credit-risk-dataset")
    file_path = os.path.join(dataset_dir,"credit_risk_dataset.csv")

    return pd.read_csv(file_path)