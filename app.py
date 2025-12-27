import streamlit as st
import pandas as pd
import numpy as np
import joblib

st.set_page_config(page_title="Bank Marketing — Decision Tree", layout="wide")

@st.cache_resource
def load_artifact(path: str):
    return joblib.load(path)

st.title("🏦 Customer Purchase Prediction — Decision Tree")
st.caption("This app loads a pre-trained model and predicts subscription. No training happens here.")

artifact = load_artifact("bank_marketing_tree.pkl")
model = artifact["model"]
feature_cols = artifact["feature_columns"]
thr = float(artifact.get("opt_threshold", 0.5))

with st.sidebar:
    st.header("Settings")
    use_saved = st.checkbox("Use saved threshold", value=True)
    threshold = thr if use_saved else st.slider("Custom threshold", 0.05, 0.95, 0.50, 0.01)

st.subheader("Upload PCA CSV")
st.write("Your CSV must contain these columns (exact):")
with st.expander("Show required PCA columns"):
    st.write(feature_cols)

uploaded = st.file_uploader("Upload CSV", type=["csv"])
if uploaded is None:
    st.info("⬆️ Upload your PCA-transformed CSV to get predictions.")
    st.stop()

df = pd.read_csv(uploaded)

missing = [c for c in feature_cols if c not in df.columns]
if missing:
    st.error(f"Missing required columns: {missing}")
    st.stop()

X = df[feature_cols].copy()
X = X.apply(pd.to_numeric, errors="coerce")

if X.isna().any().any():
    bad = X.columns[X.isna().any()].tolist()
    st.error(f"Non-numeric / missing values found in: {bad}")
    st.stop()

proba = model.predict_proba(X)[:, 1]
pred = (proba >= threshold).astype(int)

out = df.copy()
out["proba_yes"] = proba
out["pred_yes"] = pred

st.success("Predictions completed.")
st.dataframe(out.head(20), use_container_width=True)

st.download_button(
    "⬇️ Download predictions",
    data=out.to_csv(index=False).encode("utf-8"),
    file_name="tree_predictions.csv",
    mime="text/csv"
)
