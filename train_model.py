# train_model.py
import json
import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from collections import Counter
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix
from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier

# ---------- CONFIG ----------
DATA_PATH = "final_dataset.csv"   # put your CSV here
TARGET_COL = "CLINICAL_STAGE"
EXPORT_DIR = Path(".")
RANDOM_STATE = 42

# ---------- 1. Load data ----------
df = pd.read_csv(DATA_PATH)
print("Loaded dataset:", df.shape)

# ---------- 2. Basic cleaning ----------
df.replace("not available", pd.NA, inplace=True)

# Features are all columns except target
feature_cols = [c for c in df.columns if c != TARGET_COL]
print("Number of features:", len(feature_cols))

# Fill missing numeric by median, categorical by mode
num_cols = df[feature_cols].select_dtypes(include=[np.number]).columns.tolist()
cat_cols = [c for c in feature_cols if c not in num_cols]

if num_cols:
    df[num_cols] = df[num_cols].fillna(df[num_cols].median())
for c in cat_cols:
    if df[c].isnull().any():
        df[c] = df[c].fillna(df[c].mode().iloc[0])

# ---------- 3. Encode categorical features and save encoders ----------
feature_encoders = {}
df_encoded = df.copy()
for col in feature_cols:
    if df_encoded[col].dtype == "object" or str(df_encoded[col].dtype).startswith("category"):
        le = LabelEncoder()
        # convert to string first to preserve entries like G3
        df_encoded[col] = le.fit_transform(df_encoded[col].astype(str))
        feature_encoders[col] = le
        print(f"Encoded {col}: {len(le.classes_)} classes")

# ---------- 4. Encode target ----------
le_target = LabelEncoder()
df_encoded[TARGET_COL] = le_target.fit_transform(df_encoded[TARGET_COL].astype(str))
print("Target classes:", list(le_target.classes_))

# ---------- 5. Split (stratified) ----------
X = df_encoded[feature_cols]
y = df_encoded[TARGET_COL]

X_train, X_temp, y_train, y_temp = train_test_split(
    X, y, test_size=0.30, stratify=y, random_state=RANDOM_STATE
)
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.5, stratify=y_temp, random_state=RANDOM_STATE
)
print("Train/Val/Test shapes:", X_train.shape, X_val.shape, X_test.shape)

# ---------- 6. Scale ----------
scaler = StandardScaler()
X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train), columns=feature_cols, index=X_train.index)
X_val_scaled   = pd.DataFrame(scaler.transform(X_val), columns=feature_cols, index=X_val.index)
X_test_scaled  = pd.DataFrame(scaler.transform(X_test), columns=feature_cols, index=X_test.index)

# ---------- 7. Balance training set with SMOTE ----------
class_counts = Counter(y_train)
min_count = min(class_counts.values())
k_neighbors = 1 if min_count <= 2 else 5
sm = SMOTE(random_state=RANDOM_STATE, k_neighbors=k_neighbors)
X_train_bal, y_train_bal = sm.fit_resample(X_train_scaled, y_train)
print("Balanced class counts:", Counter(y_train_bal))

# ---------- 8. Train XGBoost ----------
model = XGBClassifier(
    use_label_encoder=False,
    eval_metric="mlogloss",
    n_estimators=300,
    learning_rate=0.05,
    max_depth=6,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=RANDOM_STATE
)
print("Training XGBoost...")
model.fit(X_train_bal, y_train_bal)

# ---------- 9. Evaluate ----------
print("\nValidation Results")
y_val_pred = model.predict(X_val_scaled)
print(classification_report(y_val, y_val_pred, zero_division=0))
print("Confusion matrix (val):\n", confusion_matrix(y_val, y_val_pred))

print("\nTest Results")
y_test_pred = model.predict(X_test_scaled)
print(classification_report(y_test, y_test_pred, zero_division=0))
print("Confusion matrix (test):\n", confusion_matrix(y_test, y_test_pred))

# ---------- 10. Save artifacts ----------
EXPORT_DIR.mkdir(parents=True, exist_ok=True)
joblib.dump(model, EXPORT_DIR / "xgboost_model.joblib")
joblib.dump(scaler, EXPORT_DIR / "scaler.joblib")
joblib.dump(feature_cols, EXPORT_DIR / "feature_list.joblib")
joblib.dump(feature_encoders, EXPORT_DIR / "feature_encoders.joblib")
joblib.dump(le_target, EXPORT_DIR / "target_encoder.joblib")

with open(EXPORT_DIR / "stage_classes.json", "w") as f:
    json.dump(list(le_target.classes_), f)

# Save example inputs (first 10 rows, original raw df)
df[feature_cols].iloc[:10].to_csv(EXPORT_DIR / "example_inputs.csv", index=False)

# Save X_train (for optional SHAP heavy precompute)
pd.DataFrame(X_train_scaled, columns=feature_cols).to_csv(EXPORT_DIR / "X_train_scaled_for_shap.csv", index=False)

print("\nSaved artifacts:")
print(" xgboost_model.joblib, scaler.joblib, feature_list.joblib, feature_encoders.joblib, target_encoder.joblib, stage_classes.json, example_inputs.csv")
