# train.py — FINAL (Region-specific features)

import os
import sys
import yaml
import joblib
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, f1_score

from src.models import get_models

# ---------------------------------------
# SETUP
# ---------------------------------------
MODEL_DIR = "models"
REPORT_DIR = "reports"
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(REPORT_DIR, exist_ok=True)

# ---------------------------------------
# REGION INPUT
# ---------------------------------------
if len(sys.argv) < 2:
    raise ValueError("Usage: python train.py <vizag | kadapa>")

region = sys.argv[1].lower()

# ---------------------------------------
# LOAD CONFIG
# ---------------------------------------
cfg = yaml.safe_load(open("config.yaml"))

if region not in cfg["regions"]:
    raise ValueError(f"Invalid region '{region}'")

region_cfg = cfg["regions"][region]

dataset_path = os.path.join(
    region_cfg["data_path"], "raw", f"{region}_tabular.csv"
)

if not os.path.exists(dataset_path):
    raise FileNotFoundError(f"Dataset not found: {dataset_path}")

df = pd.read_csv(dataset_path)

# ---------------------------------------
# FEATURES & TARGET
# ---------------------------------------
FEATURE_COLS = region_cfg["features"]
TARGET_COL = "label"

missing = [c for c in FEATURE_COLS + [TARGET_COL] if c not in df.columns]
if missing:
    raise ValueError(f"Missing columns: {missing}")

X = df[FEATURE_COLS]
y = df[TARGET_COL]

# ---------------------------------------
# PREPROCESSING
# ---------------------------------------
imputer = SimpleImputer(strategy="mean")
scaler = StandardScaler()

X = imputer.fit_transform(X)
X = scaler.fit_transform(X)

# ---------------------------------------
# SPLIT
# ---------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# ---------------------------------------
# TRAIN & COMPARE MODELS
# ---------------------------------------
models = get_models()
results = {}

for name, model in models.items():
    model.fit(X_train, y_train)
    preds = model.predict(X_test)

    acc = accuracy_score(y_test, preds)
    f1 = f1_score(y_test, preds)

    results[name] = {
        "model": model,
        "accuracy": acc,
        "f1": f1
    }

    joblib.dump(model, f"{MODEL_DIR}/{region}_{name}.joblib")

# ---------------------------------------
# COMPARISON PLOT
# ---------------------------------------
names = list(results.keys())
accs = [results[m]["accuracy"] for m in names]
f1s = [results[m]["f1"] for m in names]

plt.figure(figsize=(8, 5))
plt.plot(names, accs, marker="o", label="Accuracy")
plt.plot(names, f1s, marker="s", label="F1 Score")
plt.title(f"Model Comparison — {region.capitalize()}")
plt.xlabel("Model")
plt.ylabel("Score")
plt.legend()
plt.grid(True)
plt.tight_layout()

plot_path = f"{REPORT_DIR}/{region}_model_comparison.png"
plt.savefig(plot_path)
plt.close()

# ---------------------------------------
# SELECT BEST MODEL
# ---------------------------------------
best_model_name = max(results, key=lambda m: results[m]["f1"])
best_model = results[best_model_name]["model"]

# ---------------------------------------
# SAVE BUNDLE
# ---------------------------------------
joblib.dump(
    {
        "imputer": imputer,
        "scaler": scaler,
        "model": best_model,
        "best_model_name": best_model_name,
        "feature_cols": FEATURE_COLS,
        "region": region
    },
    f"{MODEL_DIR}/{region}_best_model.joblib"
)

print("\n✅ TRAINING COMPLETE")
print(f"Region      : {region}")
print(f"Best Model  : {best_model_name}")
print(f"Model File  : models/{region}_best_model.joblib")
print(f"Plot Saved  : {plot_path}")
