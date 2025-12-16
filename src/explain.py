"""
Enterprise Explainability Script
- Generates high-quality SHAP feature-importance bar chart
- Generates LIME explanation
- No beeswarm (replaces confusing scatter plot)
"""

import os
import numpy as np
import pandas as pd
import joblib
import shap
from lime.lime_tabular import LimeTabularExplainer
import matplotlib.pyplot as plt

DATA_PATH = "data/processed/processed_dataset.csv"
MODEL_PATH = "models/xgb_model.joblib"
OUTPUT_DIR = "outputs"

SHAP_FIG_PATH = os.path.join(OUTPUT_DIR, "shap_feature_importance.png")
LIME_HTML_PATH = os.path.join(OUTPUT_DIR, "lime_explanation.html")

os.makedirs(OUTPUT_DIR, exist_ok=True)


# -----------------------------------------------------------
# SAFE DATA LOADING
# -----------------------------------------------------------
def load_clean_data():
    print("[INFO] Loading dataset...")
    df = pd.read_csv(DATA_PATH)

    # Strip accidental spaces in column names
    df.columns = df.columns.str.strip()

    # Replace inf/-inf with NaN
    df = df.replace([np.inf, -np.inf], np.nan)

    # Drop rows with NaN
    df = df.dropna()

    # Drop duplicate column names
    df = df.loc[:, ~df.columns.duplicated()]

    print(f"[INFO] Clean dataset shape: {df.shape}")
    return df


# -----------------------------------------------------------
# MAIN
# -----------------------------------------------------------
def main():
    df = load_clean_data()

    print("[INFO] Loading model...")
    model = joblib.load(MODEL_PATH)

    # Identify label column
    label_col = None
    for c in ["Label", "label", "Class", "class"]:
        if c in df.columns:
            label_col = c
            break

    if label_col is None:
        raise ValueError("Dataset is missing a label column")

    X = df.drop(columns=[label_col], errors="ignore")
    y = df[label_col]

    # Ensure numeric-only features
    X = X.select_dtypes(include=[np.number])

    # -----------------------------------------------------------
    # SHAP VALUES
    # -----------------------------------------------------------
    print("[INFO] Computing SHAP values...")
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X[:200])  # limit for speed

    # -----------------------------------------------------------
    # ENTERPRISE FEATURE IMPORTANCE BAR CHART
    # -----------------------------------------------------------
    print("[INFO] Creating SHAP feature-importance bar chart...")

    importance = np.abs(shap_values).mean(axis=0)
    feature_importance = pd.DataFrame({
        "feature": X.columns,
        "importance": importance
    }).sort_values("importance", ascending=False).head(20)

    plt.figure(figsize=(10, 8))
    plt.barh(feature_importance["feature"], feature_importance["importance"],
             color=plt.cm.cool(np.linspace(0.1, 1, len(feature_importance))))
    plt.xlabel("Mean |SHAP Value|")
    plt.title("Top 20 Feature Importance (SHAP)")
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.savefig(SHAP_FIG_PATH, dpi=300)
    plt.close()

    print(f"[SUCCESS] SHAP feature-importance saved to {SHAP_FIG_PATH}")

    # -----------------------------------------------------------
    # LIME EXPLANATION
    # -----------------------------------------------------------
    print("[INFO] Generating LIME explanation...")
    explainer_lime = LimeTabularExplainer(
        training_data=np.array(X),
        mode="classification",
        feature_names=X.columns.tolist(),
        class_names=["Benign", "Attack"],
        discretize_continuous=True
    )

    row = X.iloc[5]
    exp = explainer_lime.explain_instance(
        data_row=row,
        predict_fn=model.predict_proba
    )
    exp.save_to_file(LIME_HTML_PATH)

    print(f"[SUCCESS] LIME explanation saved to {LIME_HTML_PATH}")

    # DONE
    print("\n[SUCCESS] Explainability artifacts generated!")
    print("✔ SHAP:", SHAP_FIG_PATH)
    print("✔ LIME:", LIME_HTML_PATH)


if __name__ == "__main__":
    main()
