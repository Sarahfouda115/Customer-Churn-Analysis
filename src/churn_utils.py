"""Reusable helpers for the Telco churn project (cleaning, features, preprocessing, evaluation)."""
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, roc_auc_score, average_precision_score)

RANDOM_STATE = 42
ADDON_COLS = ["OnlineSecurity", "OnlineBackup", "DeviceProtection",
              "TechSupport", "StreamingTV", "StreamingMovies"]
NUMERIC_FEATURES = ["tenure", "MonthlyCharges", "TotalCharges", "NumAddOns"]


def clean_data(raw: pd.DataFrame):
    """Return (cleaned_df, log). The raw DataFrame is never modified."""
    df = raw.copy()
    log = {}
    # Strip whitespace from every text column (catches hidden blank strings)
    for c in df.select_dtypes(include=["object", "string"]).columns:
        df[c] = df[c].str.strip()
    # TotalCharges is text in the source file; blanks become NaN on conversion
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"].replace("", np.nan), errors="coerce")
    blank = df["TotalCharges"].isna()
    log["blank_total_charges"] = int(blank.sum())
    log["blank_rows_all_tenure_zero"] = bool((df.loc[blank, "tenure"] == 0).all())
    # Customers with tenure 0 have not completed a billing cycle: TotalCharges = 0
    df.loc[blank & (df["tenure"] == 0), "TotalCharges"] = 0.0
    log["duplicate_rows"] = int(df.duplicated().sum())
    log["duplicate_ids"] = int(df["customerID"].duplicated().sum())
    return df, log


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add engineered features (no target information is used -> no leakage)."""
    out = df.copy()
    out["NumAddOns"] = (out[ADDON_COLS] == "Yes").sum(axis=1)
    return out


def split_columns(X: pd.DataFrame):
    num = [c for c in NUMERIC_FEATURES if c in X.columns]
    cat = [c for c in X.columns if c not in num]
    return num, cat


def build_preprocessor(num_cols, cat_cols) -> ColumnTransformer:
    """Scale numerics, one-hot encode categoricals. Fit only inside a Pipeline."""
    return ColumnTransformer([
        ("num", StandardScaler(), num_cols),
        ("cat", OneHotEncoder(handle_unknown="ignore", drop="if_binary"), cat_cols),
    ])


def evaluate(name, y_true, proba, threshold=0.5) -> dict:
    pred = (proba >= threshold).astype(int)
    return {"Model": name,
            "Accuracy": accuracy_score(y_true, pred),
            "Precision": precision_score(y_true, pred, zero_division=0),
            "Recall": recall_score(y_true, pred),
            "F1": f1_score(y_true, pred),
            "ROC-AUC": roc_auc_score(y_true, proba),
            "PR-AUC": average_precision_score(y_true, proba)}
