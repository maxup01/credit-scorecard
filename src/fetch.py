import os
import kagglehub
from pandas import DataFrame, pandas as pd

DATASET = "laotse/credit-risk-dataset"

def fetch_dataset() -> DataFrame:
    """Download and load the credit risk dataset.

    Downloads the Kaggle dataset ``laotse/credit-risk-dataset`` (via
    ``kagglehub``, which caches it locally so repeat calls skip the
    download) and reads its ``credit_risk_dataset.csv`` file into a
    pandas DataFrame.

    Returns:
        DataFrame: The credit risk dataset, one row per loan applicant.
    """
    dataset_dir = kagglehub.dataset_download("laotse/credit-risk-dataset")
    file_path = os.path.join(dataset_dir,"credit_risk_dataset.csv")

    return pd.read_csv(file_path)