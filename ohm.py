"""
OpenOcean ML Pipeline
=====================
Merges all 4 buoy CSV files and trains an XGBoost model.

TARGET: temperature  <-- change to: salinity, ph, dissolved_oxygen, turbidity, chlorophyll, wave_height
TASK:   regression   <-- change to "classification" for high/medium/low bucketing
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    mean_squared_error, mean_absolute_error, r2_score,
    classification_report, confusion_matrix,
)
from sklearn.pipeline import Pipeline
import xgboost as xgb
import joblib
import warnings
warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────
# CONFIG — change these as needed
# ─────────────────────────────────────────────
TARGET = "temperature"       # what to predict
TASK   = "regression"        # "regression" or "classification"
TEST_SIZE   = 0.2
RANDOM_SEED = 42

DATA_FILES = [
    "openocean_dissolved_oxygen_data.csv",
    "openocean_ph_data.csv",
    "openocean_salinity_data.csv",
    "openocean_temperature_data.csv",
]

# ─────────────────────────────────────────────
# 1. LOAD & MERGE
# ─────────────────────────────────────────────
print("=" * 55)
print("STEP 1: Loading and merging data files")
print("=" * 55)

dfs = []
for f in DATA_FILES:
    df = pd.read_csv(f)
    df["source_file"] = f.replace(".csv", "")   # track which file each row came from
    dfs.append(df)

data = pd.concat(dfs, ignore_index=True)
print(f"Total rows after merge : {len(data):,}")
print(f"Columns                : {list(data.columns)}")
print(f"Duplicates             : {data.duplicated().sum():,}")

# Drop exact duplicates — same buoy+timestamp can appear across multiple files
data.drop_duplicates(inplace=True)
print(f"Rows after dedup       : {len(data):,}\n")

# ─────────────────────────────────────────────
# 2. FEATURE ENGINEERING
# ─────────────────────────────────────────────
print("=" * 55)
print("STEP 2: Feature engineering")
print("=" * 55)

data["timestamp"] = pd.to_datetime(data["timestamp"], utc=True)

# Basic time features
data["hour"]        = data["timestamp"].dt.hour
data["day_of_week"] = data["timestamp"].dt.dayofweek
data["month"]       = data["timestamp"].dt.month
data["year"]        = data["timestamp"].dt.year

# Cyclical encoding so hour 23 and hour 0 are "close" to the model
data["hour_sin"]  = np.sin(2 * np.pi * data["hour"]  / 24)
data["hour_cos"]  = np.cos(2 * np.pi * data["hour"]  / 24)
data["month_sin"] = np.sin(2 * np.pi * data["month"] / 12)
data["month_cos"] = np.cos(2 * np.pi * data["month"] / 12)

# Encode buoy_id as integer so the model can use it
data["buoy_id_enc"] = data["buoy_id"].astype("category").cat.codes

print("New features: hour, day_of_week, month, year,")
print("              hour_sin/cos, month_sin/cos, buoy_id_enc\n")

# ─────────────────────────────────────────────
# 3. SELECT FEATURES & TARGET
# ─────────────────────────────────────────────
FEATURE_COLS = [
    "latitude", "longitude",
    "salinity", "ph", "dissolved_oxygen",
    "turbidity", "chlorophyll", "wave_height",
    "hour_sin", "hour_cos",
    "month_sin", "month_cos",
    "day_of_week", "year",
    "buoy_id_enc",
]

# Make sure target isn't also a feature (data leakage)
FEATURE_COLS = [c for c in FEATURE_COLS if c != TARGET]

print("=" * 55)
print("STEP 3: Preparing features and target")
print("=" * 55)
print(f"Target   : {TARGET}")
print(f"Features : {FEATURE_COLS}\n")

df_model = data[FEATURE_COLS + [TARGET]].dropna()
print(f"Usable rows (no nulls): {len(df_model):,}\n")

X = df_model[FEATURE_COLS]
y = df_model[TARGET]

# For classification: bin continuous target into 3 equal-frequency buckets
if TASK == "classification":
    y, bins = pd.qcut(y, q=3, labels=["low", "medium", "high"], retbins=True)
    print(f"Classification bins: low < {bins[1]:.2f} <= medium < {bins[2]:.2f} <= high\n")

# ─────────────────────────────────────────────
# 4. TRAIN / TEST SPLIT
# ─────────────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=TEST_SIZE, random_state=RANDOM_SEED
)
print(f"Train size : {len(X_train):,}")
print(f"Test size  : {len(X_test):,}\n")

# ─────────────────────────────────────────────
# 5. BUILD MODEL PIPELINE
# ─────────────────────────────────────────────
print("=" * 55)
print("STEP 4: Building XGBoost pipeline")
print("=" * 55)

if TASK == "regression":
    xgb_model = xgb.XGBRegressor(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=RANDOM_SEED,
        verbosity=0,
    )
else:
    xgb_model = xgb.XGBClassifier(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        eval_metric="mlogloss",
        random_state=RANDOM_SEED,
        verbosity=0,
    )

# StandardScaler + XGBoost in one clean pipeline
pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("model",  xgb_model),
])

# ─────────────────────────────────────────────
# 6. TRAIN
# ─────────────────────────────────────────────
print("Training model...")
pipeline.fit(X_train, y_train)
print("Done.\n")

# ─────────────────────────────────────────────
# 7. EVALUATE
# ─────────────────────────────────────────────
print("=" * 55)
print("STEP 5: Evaluation on test set")
print("=" * 55)

y_pred = pipeline.predict(X_test)

if TASK == "regression":
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae  = mean_absolute_error(y_test, y_pred)
    r2   = r2_score(y_test, y_pred)
    print(f"RMSE : {rmse:.4f}")
    print(f"MAE  : {mae:.4f}")
    print(f"R²   : {r2:.4f}\n")
else:
    print(classification_report(y_test, y_pred))
    r2 = None   # not used in classification plots

# ─────────────────────────────────────────────
# 8. PLOTS
# ─────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle(
    f"OpenOcean XGBoost  —  predicting '{TARGET}'  ({TASK})",
    fontsize=13, fontweight="bold"
)

# Plot 1: Feature Importance
feat_imp = pd.Series(
    pipeline.named_steps["model"].feature_importances_,
    index=FEATURE_COLS
).sort_values(ascending=True)

feat_imp.plot(kind="barh", ax=axes[0], color="steelblue")
axes[0].set_title("Feature Importance")
axes[0].set_xlabel("Gain Score")

# Plot 2: Actual vs Predicted  OR  Confusion Matrix
if TASK == "regression":
    axes[1].scatter(y_test, y_pred, alpha=0.25, s=8, color="steelblue")
    lo = min(float(y_test.min()), float(y_pred.min()))
    hi = max(float(y_test.max()), float(y_pred.max()))
    axes[1].plot([lo, hi], [lo, hi], "r--", linewidth=1.5, label="Perfect fit")
    axes[1].set_xlabel(f"Actual {TARGET}")
    axes[1].set_ylabel(f"Predicted {TARGET}")
    axes[1].set_title(f"Actual vs Predicted  |  R² = {r2:.3f}")
    axes[1].legend()
else:
    labels = ["low", "medium", "high"]
    cm = confusion_matrix(y_test, y_pred, labels=labels)
    sns.heatmap(
        cm, annot=True, fmt="d", ax=axes[1],
        xticklabels=labels, yticklabels=labels, cmap="Blues"
    )
    axes[1].set_title("Confusion Matrix")
    axes[1].set_xlabel("Predicted")
    axes[1].set_ylabel("Actual")

plt.tight_layout()
plt.savefig("openocean_results.png", dpi=150, bbox_inches="tight")
print("Plot saved → openocean_results.png\n")

# ─────────────────────────────────────────────
# 9. SAVE MODEL
# ─────────────────────────────────────────────
joblib.dump(pipeline, "openocean_model.pkl")
print("Model saved  openocean_model.pkl")
print("\nTo reload and predict later:")
print("  import joblib, pandas as pd")
print("  model = joblib.load('openocean_model.pkl')")
print("  preds = model.predict(new_df[FEATURE_COLS])\n")

# ─────────────────────────────────────────────
# 10. SAMPLE PREDICTIONS
# ─────────────────────────────────────────────
print("=" * 55)
print("STEP 6: Sample predictions on 5 test rows")
print("=" * 55)
sample = X_test.head(5).copy()
sample["actual"]    = list(y_test.head(5))
sample["predicted"] = list(pipeline.predict(X_test.head(5)))
print(sample[["actual", "predicted"]].to_string())
print("\nAll done!")