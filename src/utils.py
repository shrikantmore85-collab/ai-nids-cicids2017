import os
import joblib

def ensure_directories(dirs):
    for d in dirs:
        if not os.path.exists(d):
            os.makedirs(d, exist_ok=True)
            print(f"[INFO] Created directory: {d}")

def save_label_encoder(le, path):
    joblib.dump(le, path)
    print(f"[INFO] Saved label encoder to {path}")

def load_label_encoder(path):
    if os.path.exists(path):
        return joblib.load(path)
    return None
