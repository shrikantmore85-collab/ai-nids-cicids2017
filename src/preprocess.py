import os
import glob
import pandas as pd
import numpy as np

RAW_DIR = "data/raw/"
OUT_PATH = "data/processed/processed_dataset.csv"


def load_and_merge_raw_files(raw_dir=RAW_DIR):
    csv_files = glob.glob(os.path.join(raw_dir, "*.csv"))

    if len(csv_files) == 0:
        raise FileNotFoundError("No CSV files found in data/raw/")

    print(f"[INFO] Found {len(csv_files)} CSV files.")
    
    df_list = []
    for f in csv_files:
        try:
            df = pd.read_csv(f, low_memory=False)
            print(f"[INFO] Loaded: {os.path.basename(f)} shape={df.shape}")
            df_list.append(df)
        except Exception as e:
            print(f"[WARN] Could not load {f}: {e}")

    df = pd.concat(df_list, ignore_index=True)
    print(f"[INFO] Merged dataset shape: {df.shape}")
    return df


def clean_column_names(df):
    df.columns = (
        df.columns
        .str.strip()
        .str.replace(" ", "_")
        .str.replace("/", "_")
        .str.replace(".", "_")
        .str.replace("-", "_")
    )
    return df


def ensure_label_column(df):
    possible_labels = ["Label", "label", "Attack", "attack_type"]

    found = None
    for col in possible_labels:
        if col in df.columns:
            found = col
            break

    if found is None:
        raise ValueError("Dataset does not contain a label column.")

    df = df.rename(columns={found: "label"})
    return df


def convert_label_to_binary(df):
    df["label"] = df["label"].astype(str).str.upper()

    df["label"] = df["label"].apply(lambda x: 0 if x == "BENIGN" else 1)
    return df


def convert_numeric(df):
    for col in df.columns:
        if col == "label":
            continue
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def clean_dataframe(df):
    df = clean_column_names(df)

    df = ensure_label_column(df)

    df = convert_label_to_binary(df)

    df = convert_numeric(df)

    # Remove rows where label=nan
    df = df.dropna(subset=["label"])

    # Fill remaining NaN with column means
    df = df.fillna(df.mean(numeric_only=True))

    print(f"[INFO] Final cleaned shape: {df.shape}")
    return df


def main():
    print("[INFO] Starting preprocessing...")
    
    df = load_and_merge_raw_files()

    df = clean_dataframe(df)

    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    df.to_csv(OUT_PATH, index=False)

    print(f"[INFO] Saved processed dataset to {OUT_PATH}")


if __name__ == "__main__":
    main()
