"""Load the raw Telco file, clean it, and create the train/test split.

Usage (from project root):
    python src/data_preprocessing.py
Writes data/processed/telco_churn_cleaned.csv (raw file is never modified).
"""
from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split
import churn_utils as cu

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw" / "Telco-Customer-Churn.csv"
PROCESSED = ROOT / "data" / "processed" / "telco_churn_cleaned.csv"


def load_and_clean(raw_path=RAW):
    raw = pd.read_csv(raw_path)
    df, log = cu.clean_data(raw)
    df["Churn_flag"] = (df["Churn"] == "Yes").astype(int)
    return df, log


def make_split(df, test_size=0.20):
    """Same stratified split as the notebook (random_state=42). Returns X_train, X_test, y_train, y_test."""
    model_df = cu.add_features(df)
    y = model_df["Churn_flag"]
    X = model_df.drop(columns=["customerID", "Churn", "Churn_flag"])
    return train_test_split(X, y, test_size=test_size, stratify=y, random_state=cu.RANDOM_STATE)


if __name__ == "__main__":
    df, log = load_and_clean()
    PROCESSED.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(PROCESSED, index=False)
    print("Cleaning log:", log)
    print(f"Saved {PROCESSED.relative_to(ROOT)} {df.shape}")
