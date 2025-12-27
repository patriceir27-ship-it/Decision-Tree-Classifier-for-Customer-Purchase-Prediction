import numpy as np
import pandas as pd
import joblib

def load_artifact(path: str) -> dict:
    art = joblib.load(path)
    required = {"model", "feature_columns"}
    missing = required - set(art.keys())
    if missing:
        raise ValueError(f"Artifact missing keys: {missing}")
    if "opt_threshold" not in art:
        art["opt_threshold"] = 0.5
    return art

def validate_features(df: pd.DataFrame, feature_cols: list) -> pd.DataFrame:
    missing = [c for c in feature_cols if c not in df.columns]
    extra = [c for c in df.columns if c not in feature_cols]
    if missing:
        raise ValueError(f"Missing required PCA columns: {missing[:10]}{'...' if len(missing)>10 else ''}")
    df = df[feature_cols].copy()
    df = df.apply(pd.to_numeric, errors="coerce")
    if df.isna().any().any():
        bad_cols = df.columns[df.isna().any()].tolist()
        raise ValueError(f"Non-numeric / missing values found in columns: {bad_cols[:10]}{'...' if len(bad_cols)>10 else ''}")
    return df

def predict_proba_and_label(model, X: pd.DataFrame, threshold: float):
    proba = model.predict_proba(X)[:, 1]
    pred = (proba >= threshold).astype(int)
    return proba, pred

def make_template(feature_cols: list, n_rows: int = 5) -> pd.DataFrame:
    return pd.DataFrame(np.zeros((n_rows, len(feature_cols))), columns=feature_cols)
