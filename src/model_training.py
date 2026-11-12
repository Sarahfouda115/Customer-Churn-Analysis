"""Train and compare the six classifiers, then save the comparison table and the chosen model.

Usage (from project root):   python src/model_training.py [--out-dir outputs] [--model-dir models]
Mirrors Phase 6-7 of the notebook (same split, grids, CV and seed).
"""
import argparse, time, warnings
from pathlib import Path
import joblib, numpy as np, pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import fbeta_score, roc_auc_score
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_val_predict
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
import churn_utils as cu
import data_preprocessing as dp

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[1]
FINAL_MODEL = "Logistic Regression"   # chosen in the notebook (Phase 7): tied on AUC, most stable, most explainable


def candidates(rs=cu.RANDOM_STATE):
    return {
        "Logistic Regression": (LogisticRegression(max_iter=3000, class_weight="balanced"), {"clf__C": [0.01, 0.1, 1, 10, 100]}),
        "Decision Tree": (DecisionTreeClassifier(class_weight="balanced", random_state=rs),
                          {"clf__max_depth": [3, 5, 7], "clf__min_samples_leaf": [20, 50, 100]}),
        "Random Forest": (RandomForestClassifier(class_weight="balanced", random_state=rs, n_jobs=-1),
                          {"clf__n_estimators": [300], "clf__max_depth": [6, 10, None], "clf__min_samples_leaf": [5, 20]}),
        "KNN": (KNeighborsClassifier(), {"clf__n_neighbors": [15, 31, 61, 91], "clf__weights": ["uniform", "distance"]}),
        "SVM (RBF)": (SVC(probability=True, class_weight="balanced", random_state=rs), {"clf__C": [0.1, 1], "clf__gamma": ["scale", 0.01]}),
        "Gradient Boosting": (GradientBoostingClassifier(random_state=rs),
                              {"clf__n_estimators": [100, 200], "clf__learning_rate": [0.05, 0.1], "clf__max_depth": [2, 3]}),
    }


def main(out_dir, model_dir):
    df, _ = dp.load_and_clean()
    X_train, X_test, y_train, y_test = dp.make_split(df)
    num, cat = cu.split_columns(X_train)
    cv = StratifiedKFold(5, shuffle=True, random_state=cu.RANDOM_STATE)
    mk = lambda est: Pipeline([("prep", cu.build_preprocessor(num, cat)), ("clf", est)])

    rows, fitted = [], {}
    base = mk(DummyClassifier(strategy="prior")).fit(X_train, y_train)
    rows.append(cu.evaluate("Baseline (always 'No churn')", y_test, base.predict_proba(X_test)[:, 1]))
    for name, (est, grid) in candidates().items():
        t0 = time.time()
        gs = GridSearchCV(mk(est), grid, scoring="roc_auc", cv=cv, n_jobs=-1).fit(X_train, y_train)
        fitted[name] = gs.best_estimator_
        r = cu.evaluate(name, y_test, gs.predict_proba(X_test)[:, 1]); r["CV AUC"] = gs.best_score_
        rows.append(r); print(f"{name:20s} CV AUC {gs.best_score_:.4f}  test AUC {r['ROC-AUC']:.4f}  ({time.time()-t0:.0f}s)")

    out_dir.mkdir(parents=True, exist_ok=True); model_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).set_index("Model").round(4).to_csv(out_dir / "model_comparison.csv")

    # threshold chosen on out-of-fold TRAINING predictions only (max F2), as in the notebook
    final = fitted[FINAL_MODEL]
    oof = cross_val_predict(final, X_train, y_train, cv=cv, method="predict_proba", n_jobs=-1)[:, 1]
    grid_t = np.round(np.arange(0.20, 0.81, 0.05), 2)
    best_t = float(grid_t[np.argmax([fbeta_score(y_train, oof >= t, beta=2) for t in grid_t])])
    joblib.dump({"pipeline": final, "threshold": best_t, "model_name": FINAL_MODEL, "features": list(X_train.columns)},
                model_dir / "best_model.joblib")
    print(f"Saved model ({FINAL_MODEL}, threshold {best_t}) and comparison table.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", type=Path, default=ROOT / "outputs")
    ap.add_argument("--model-dir", type=Path, default=ROOT / "models")
    a = ap.parse_args(); main(a.out_dir, a.model_dir)
