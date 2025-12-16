"""
train.py
- Reads all CSV files from data/processed/
- Concatenates into a single DataFrame
- Cleans NaN/inf values
- Encodes textual labels to integers (saves LabelEncoder)
- Trains an XGBoost classifier (or a fallback RandomForest if XGBoost not available)
- Saves trained model to models/xgb_model.joblib and label encoder to models/label_encoder.joblib
- Prints training metrics
"""

import os
import glob
import sys
import warnings
from typing import Tuple

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report
import joblib

# Prefer XGBoost; fallback to RandomForest if not installed
try:
    from xgboost import XGBClassifier
    _USE_XGB = True
except Exception:
    from sklearn.ensemble import RandomForestClassifier
    _USE_XGB = False

# Paths
PROCESSED_GLOB = "data/processed/*.csv"
MODEL_DIR = "models"
MODEL_PATH = os.path.join(MODEL_DIR, "xgb_model.joblib")
LABEL_ENCODER_PATH = os.path.join(MODEL_DIR, "label_encoder.joblib")

# Ensure dirs exist
os.makedirs(MODEL_DIR, exist_ok=True)


def read_all_processed() -> pd.DataFrame:
    files = sorted(glob.glob(PROCESSED_GLOB))
    if not files:
        raise FileNotFoundError(f"No processed CSV files found in data/processed/ (expected {PROCESSED_GLOB})")
    dfs = []
    print(f"[INFO] Found {len(files)} processed file(s).")
    for f in files:
        print(f"[INFO] Loading {f} ...")
        df = pd.read_csv(f)
        dfs.append(df)
    combined = pd.concat(dfs, ignore_index=True)
    print(f"[INFO] Combined dataframe shape: {combined.shape}")
    return combined


def locate_label_column(df: pd.DataFrame) -> str:
    # Common label names (case-insensitive)
    candidates = [c for c in df.columns if c.lower() in ("label", "labels", "class", "target")]
    if candidates:
        return candidates[0]
    # try last column as fallback
    last_col = df.columns[-1]
    # perform a heuristic: many dataset label columns contain non-numeric strings like 'BENIGN' or 'DDoS'
    if df[last_col].dtype == object or df[last_col].nunique() < max(50, len(df) * 0.5):
        return last_col
    raise ValueError("Could not locate a label column. Expected a column named 'label' or similar.")


def clean_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    # Replace inf with NaN, then drop or impute. For simplicity drop rows with NaN
    df = df.replace([np.inf, -np.inf], np.nan)
    n_before = len(df)
    df = df.dropna()
    n_after = len(df)
    if n_after < n_before:
        print(f"[INFO] Dropped {n_before - n_after} rows due to NaN/inf.")
    return df


def prepare_features_labels(df: pd.DataFrame, label_col: str) -> Tuple[pd.DataFrame, np.ndarray, LabelEncoder]:
    if label_col not in df.columns:
        raise KeyError(f"Label column '{label_col}' not in dataframe")

    y_raw = df[label_col].astype(object)

    # If labels are numeric-like but stored as strings, attempt conversion
    try:
        # if all values can be integer-cast -> keep numeric
        _ = y_raw.map(lambda x: int(float(x)))
        all_numeric_like = True
    except Exception:
        all_numeric_like = False

    if all_numeric_like:
        y = y_raw.astype(int).values
        label_enc = None
        print("[INFO] Label column appears numeric; no LabelEncoder used.")
    else:
        label_enc = LabelEncoder()
        y = label_enc.fit_transform(y_raw)
        print(f"[INFO] Encoded labels: {list(label_enc.classes_)}")
    # Drop label column from features
    X = df.drop(columns=[label_col], errors="ignore")
    # Ensure features are numeric; drop non-numeric columns
    non_numeric = X.select_dtypes(exclude=[np.number]).columns.tolist()
    if non_numeric:
        print(f"[INFO] Dropping non-numeric feature columns: {non_numeric}")
        X = X.drop(columns=non_numeric)

    # Final NaN/inf check on X
    X = X.replace([np.inf, -np.inf], np.nan).dropna(axis=0)
    return X, y, label_enc


def train_model(X_train, y_train):
    if _USE_XGB:
        print("[INFO] Using XGBoost classifier")
        model = XGBClassifier(
            n_estimators=200,
            max_depth=6,
            learning_rate=0.1,
            use_label_encoder=False,
            eval_metric="logloss",
            verbosity=0,
            n_jobs=4
        )
    else:
        print("[INFO] XGBoost not available; using RandomForestClassifier fallback")
        from sklearn.ensemble import RandomForestClassifier
        model = RandomForestClassifier(n_estimators=200, n_jobs=4)
    model.fit(X_train, y_train)
    return model


def main():
    print("[INFO] Training started...")
    df = read_all_processed()
    df = clean_dataframe(df)

    # locate label column
    label_col = locate_label_column(df)
    print(f"[INFO] Using label column: {label_col}")

    X, y, label_enc = prepare_features_labels(df, label_col)
    if len(X) == 0:
        raise ValueError("No feature rows remaining after cleaning. Check your processed data.")

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    model = train_model(X_train, y_train)

    # Evaluate
    preds = model.predict(X_test)
    acc = accuracy_score(y_test, preds)
    print(f"[SUCCESS] Training complete. Test accuracy: {acc:.4f}")
    print("[INFO] Classification report:")
    print(classification_report(y_test, preds, digits=4))

    # Save model and label encoder
    joblib.dump(model, MODEL_PATH)
    print(f"[SUCCESS] Model saved to {MODEL_PATH}")
    if label_enc is not None:
        joblib.dump(label_enc, LABEL_ENCODER_PATH)
        print(f"[SUCCESS] LabelEncoder saved to {LABEL_ENCODER_PATH}")
    else:
        # remove existing encoder if present (labels numeric)
        if os.path.exists(LABEL_ENCODER_PATH):
            os.remove(LABEL_ENCODER_PATH)


if __name__ == "__main__":
    main()
