import streamlit as st
import pandas as pd

from predictor import load_artifact, validate_features, predict_proba_and_label, make_template

st.set_page_config(page_title="Bank Marketing — Decision Tree", layout="wide")

st.title("Bank Marketing — Decision Tree Deployment")
st.write("This app uses **PCA features** (PC1..PCk) exactly as trained.")

MODEL_PATH = "bank_marketing_tree.pkl"

with st.sidebar:
    st.header("Model")
    st.code(MODEL_PATH)
    st.header("Threshold")
    threshold_mode = st.radio("Choose threshold", ["Use saved threshold", "Set custom threshold"], index=0)

artifact = load_artifact(MODEL_PATH)
model = artifact["model"]
feature_cols = artifact["feature_columns"]
saved_thr = float(artifact.get("opt_threshold", 0.5))
thr = saved_thr if threshold_mode == "Use saved threshold" else st.sidebar.slider("Custom threshold", 0.05, 0.95, 0.50, 0.01)

st.subheader(" Required input schema (PCA columns)")
st.write(f"Number of PCA features expected: **{len(feature_cols)}**")
with st.expander("Show required PCA columns"):
    st.write(feature_cols)

colA, colB = st.columns(2)

with colA:
    st.subheader("1) Single Prediction (manual PCA input)")
    st.caption("Enter values for PCA columns. (Tip: use CSV upload for faster work.)")
    sample = pd.DataFrame({c: [0.0] for c in feature_cols})
    edited = st.data_editor(sample, use_container_width=True, num_rows="fixed")
    if st.button("Predict (Single)", type="primary"):
        X = validate_features(edited, feature_cols)
        proba, pred = predict_proba_and_label(model, X, thr)
        st.success(f"Predicted class: {'YES (subscribe)' if pred[0]==1 else 'NO'}")
        st.metric("Probability of YES", f"{proba[0]:.4f}")
        st.metric("Threshold used", f"{thr:.4f}")

with colB:
    st.subheader("2) Batch Prediction (upload PCA CSV)")
    st.caption("Upload a CSV containing the PCA columns. Output will add probability + prediction.")
    template = make_template(feature_cols, n_rows=5)
    st.download_button(
        "⬇️ Download PCA Template CSV",
        data=template.to_csv(index=False).encode("utf-8"),
        file_name="pca_template.csv",
        mime="text/csv"
    )

    file = st.file_uploader("Upload CSV", type=["csv"])
    if file is not None:
        data = pd.read_csv(file)
        st.write("Preview:")
        st.dataframe(data.head(10), use_container_width=True)

        if st.button("Predict (Batch)", type="primary"):
            X = validate_features(data, feature_cols)
            proba, pred = predict_proba_and_label(model, X, thr)
            out = data.copy()
            out["proba_yes"] = proba
            out["pred_yes"] = pred
            st.success("Done.")
            st.dataframe(out.head(20), use_container_width=True)
            st.download_button(
                "⬇️ Download Predictions CSV",
                data=out.to_csv(index=False).encode("utf-8"),
                file_name="tree_predictions.csv",
                mime="text/csv"
            )

st.divider()
st.info(
    "Important: This deployment expects **PCA-transformed inputs**.\n"
    "If you want deployment from RAW bank features (age, job, etc.), you must also save and load the full preprocessing pipeline."
)
