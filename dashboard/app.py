"""Interactive churn dashboard.   Run from the project root:   streamlit run dashboard/app.py"""
import sys
from pathlib import Path
import joblib, pandas as pd, streamlit as st

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT / "src"))
import churn_utils as cu
import prediction as pr

st.set_page_config(page_title="Telco Churn Dashboard", page_icon="📡", layout="wide")


@st.cache_data
def load():
    return (pd.read_csv(ROOT / "data" / "processed" / "telco_churn_cleaned.csv"),
            pd.read_csv(ROOT / "outputs" / "model_comparison.csv"),
            pd.read_csv(ROOT / "outputs" / "predictions.csv"))


@st.cache_resource
def model():
    return joblib.load(ROOT / "models" / "best_model.joblib")


df, comp, preds = load()
bundle = model()
overall = df["Churn_flag"].mean()
df["TenureGroup"] = pd.cut(df["tenure"], [-1, 12, 24, 48, 72], labels=["0-12", "13-24", "25-48", "49-72"])

st.title("📡 Telco Churn Dashboard")
st.caption("Associations in a historical sample, not proof of cause. Decision support, not automated decisions.")
tab1, tab2, tab3, tab4 = st.tabs(["Segment explorer", "Model scores", "Customer risk list", "What-if scoring"])

with tab1:
    c = st.columns(4)
    sel = {}
    for col, k in zip(["Contract", "InternetService", "PaymentMethod", "TenureGroup"], c):
        opts = ["All"] + [str(x) for x in sorted(df[col].dropna().unique(), key=str)]
        sel[col] = k.selectbox(col, opts)
    f = df.copy()
    for col, v in sel.items():
        if v != "All":
            f = f[f[col].astype(str) == v]
    n, k = len(f), int(f["Churn_flag"].sum())
    m = st.columns(4)
    m[0].metric("Customers", f"{n:,}"); m[1].metric("Churned", f"{k:,}")
    m[2].metric("Churn rate", f"{(k / n if n else 0):.1%}"); m[3].metric("vs overall", f"{((k / n if n else 0) - overall) * 100:+.1f} pts")
    g = st.columns(2)
    for i, col in enumerate(["Contract", "TenureGroup", "InternetService", "PaymentMethod"]):
        with g[i % 2]:
            st.subheader(f"Churn rate by {col}")
            st.bar_chart(f.groupby(col, observed=True)["Churn_flag"].mean().mul(100).rename("Churn rate (%)"))

with tab2:
    metric = st.radio("Metric", ["ROC-AUC", "PR-AUC", "Recall", "Precision", "F1", "Accuracy", "CV AUC"], horizontal=True)
    st.bar_chart(comp.set_index("Model")[metric].sort_values(ascending=False))
    st.dataframe(comp.set_index("Model").round(3), width="stretch")
    st.info(f"Selected model: **{bundle['model_name']}** (threshold {bundle['threshold']}). The top three models are statistically tied on AUC; "
            "it was chosen for stability and explainability, not accuracy. Metrics are on the 1,409-customer test set at threshold 0.50.")

with tab3:
    split = st.radio("Customers", ["Held-out test set (honest)", "All customers"], horizontal=True)
    p = preds[preds["dataset_split"] == "test"] if split.startswith("Held") else preds
    tiers = st.multiselect("Risk tier", ["High", "Medium", "Low"], default=["High"])
    p = p[p["risk_tier"].isin(tiers)].sort_values("churn_probability", ascending=False)
    st.caption("Tiers: High >= 0.75, Medium >= 0.35, Low below. Rows in the training split were used to fit the model, so their scores look better than real life.")
    if split.startswith("Held"):
        r = preds[preds["dataset_split"] == "test"].groupby("risk_tier")["actual_churn"].agg(["mean", "size"]).rename(columns={"mean": "actual churn rate", "size": "customers"})
        st.dataframe(r.style.format({"actual churn rate": "{:.1%}"}), width="content")
    st.dataframe(p, width="stretch", height=380)
    st.download_button("Download this list (CSV)", p.to_csv(index=False), "risk_list.csv")

with tab4:
    st.write("Pick a customer and change a few attributes to see how the model's risk score moves. "
             "This shows what the **model** does, not what would **cause** a real customer to stay.")
    base_id = st.selectbox("Base customer", df["customerID"].head(300))
    row = df[df["customerID"] == base_id].iloc[0].copy()
    a, b, c2 = st.columns(3)
    row["Contract"] = a.selectbox("Contract", ["Month-to-month", "One year", "Two year"], ["Month-to-month", "One year", "Two year"].index(row["Contract"]))
    row["InternetService"] = a.selectbox("Internet", ["DSL", "Fiber optic", "No"], ["DSL", "Fiber optic", "No"].index(row["InternetService"]))
    pm = list(df["PaymentMethod"].unique())
    row["PaymentMethod"] = b.selectbox("Payment", pm, pm.index(row["PaymentMethod"]))
    row["tenure"] = b.slider("Tenure (months)", 0, 72, int(row["tenure"]))
    row["MonthlyCharges"] = c2.slider("Monthly charges ($)", 18.0, 119.0, float(row["MonthlyCharges"]))
    for s in ["OnlineSecurity", "TechSupport"]:
        row[s] = c2.selectbox(s, ["Yes", "No", "No internet service"], ["Yes", "No", "No internet service"].index(row[s]))
    row["TotalCharges"] = row["tenure"] * row["MonthlyCharges"]   # simplification for the what-if
    one = pr.score(pd.DataFrame([row.drop(["Churn_flag", "TenureGroup"], errors="ignore")]), bundle).iloc[0]
    st.metric("Model risk score (not a calibrated probability)", f"{one['churn_probability']:.2f}", f"tier: {one['risk_tier']}")
