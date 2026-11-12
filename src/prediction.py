"""Score customers with the saved model.

Usage (from project root):
    python src/prediction.py                       # score every customer in the sample -> outputs/predictions.csv
    python src/prediction.py --input new.csv --output scored.csv   # score a new file (same columns as the raw Telco file)

Columns written: customerID, churn_probability, risk_flag, risk_tier, plus actual_churn and dataset_split when known.
risk_tier: High >= 0.75 (about the riskiest 20% in out-of-fold training data), Medium >= chosen threshold, Low otherwise.
Scores are for ranking customers; they are NOT calibrated probabilities (class weights were used).
Rows marked dataset_split = 'train' were used to fit the model, so only 'test' rows give honest performance.
"""
import argparse
from pathlib import Path
import joblib, pandas as pd
import churn_utils as cu
import data_preprocessing as dp

ROOT = Path(__file__).resolve().parents[1]
HIGH = 0.75


def score(raw: pd.DataFrame, bundle):
    df, _ = cu.clean_data(raw)
    X = cu.add_features(df).drop(columns=["customerID", "Churn", "Churn_flag"], errors="ignore")
    X = X[bundle["features"]]
    p = bundle["pipeline"].predict_proba(X)[:, 1]
    out = pd.DataFrame({"customerID": df["customerID"], "churn_probability": p.round(4)})
    out["risk_flag"] = (p >= bundle["threshold"]).astype(int)
    out["risk_tier"] = pd.cut(p, [-1, bundle["threshold"] - 1e-9, HIGH - 1e-9, 2], labels=["Low", "Medium", "High"]).astype(str)
    if "Churn" in df.columns:
        out["actual_churn"] = (df["Churn"] == "Yes").astype(int)
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", type=Path, default=dp.RAW)
    ap.add_argument("--output", type=Path, default=ROOT / "outputs" / "predictions.csv")
    ap.add_argument("--model", type=Path, default=ROOT / "models" / "best_model.joblib")
    a = ap.parse_args()
    bundle = joblib.load(a.model)
    res = score(pd.read_csv(a.input), bundle)
    if a.input == dp.RAW:   # mark which rows were held out in the notebook's split
        df, _ = dp.load_and_clean(); _, X_te, _, _ = dp.make_split(df)
        res["dataset_split"] = ["test" if i in set(X_te.index) else "train" for i in res.index]
    res.to_csv(a.output, index=False)
    print(f"Saved {a.output} ({len(res):,} rows) | tiers:\n{res['risk_tier'].value_counts().to_string()}")
