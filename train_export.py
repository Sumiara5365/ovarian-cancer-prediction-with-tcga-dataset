# train_export.py
import json
import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix
from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier

# ---- CONFIG ----
DATA_PATH = "final_dataset.csv"   # <- your CSV file (change if needed)
TARGET_COL = "CLINICAL_STAGE"
EXPORT_DIR = Path(".")
RANDOM_STATE = 42

# ---- 1. Load data ----
df = pd.read_csv(DATA_PATH)
print("Loaded dataset shape:", df.shape)

# ---- 2. Basic clean ----
# Replace 'not available' strings to NaN
df.replace("not available", pd.NA, inplace=True)

# Identify feature columns (all except target)
feature_cols = [c for c in df.columns if c != TARGET_COL]
print(f"Number of features (will be used for training): {len(feature_cols)}")

# Fill missing numerics by median and categorical by mode (simple but effective)
num_cols = df[feature_cols].select_dtypes(include=[np.number]).columns.tolist()
cat_cols = [c for c in feature_cols if c not in num_cols]

if num_cols:
    df[num_cols] = df[num_cols].fillna(df[num_cols].median())
if cat_cols:
    # fill categorical missing with mode
    for c in cat_cols:
        if df[c].isnull().any():
            df[c] = df[c].fillna(df[c].mode().iloc[0])

# ---- 3. Encode categorical features (GRADE etc.) ----
feature_encoders = {}
for col in feature_cols:
    if df[col].dtype == "object" or str(df[col].dtype).startswith("category"):
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))
        feature_encoders[col] = le
        print(f"Encoded {col} with {len(le.classes_)} classes")

# Encode target
le_target = LabelEncoder()
df[TARGET_COL] = le_target.fit_transform(df[TARGET_COL].astype(str))
print("Target classes:", list(le_target.classes_))

# ---- 4. Train/val/test split (stratified) ----
X = df[feature_cols]
y = df[TARGET_COL]

X_train, X_temp, y_train, y_temp = train_test_split(
    X, y, test_size=0.3, stratify=y, random_state=RANDOM_STATE
)
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.5, stratify=y_temp, random_state=RANDOM_STATE
)

print("Train/Val/Test shapes:", X_train.shape, X_val.shape, X_test.shape)

# ---- 5. Scale numeric features ----
scaler = StandardScaler()
X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train), columns=feature_cols, index=X_train.index)
X_val_scaled   = pd.DataFrame(scaler.transform(X_val), columns=feature_cols, index=X_val.index)
X_test_scaled  = pd.DataFrame(scaler.transform(X_test), columns=feature_cols, index=X_test.index)

# ---- 6. Balance training set with SMOTE ----
# choose k_neighbors safely: if smallest class has n samples, k_neighbors <= n-1
from collections import Counter
class_counts = Counter(y_train)
min_count = min(class_counts.values())
k_neighbors = 1 if min_count <= 2 else 5
print("SMOTE k_neighbors chosen:", k_neighbors)

sm = SMOTE(random_state=RANDOM_STATE, k_neighbors=k_neighbors)
X_train_bal, y_train_bal = sm.fit_resample(X_train_scaled, y_train)
print("Balanced counts:", Counter(y_train_bal))

# ---- 7. Train XGBoost (multiclass) ----
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

# ---- 8. Evaluate on validation and test sets ----
print("\nValidation results:")
y_val_pred = model.predict(X_val_scaled)
print(classification_report(y_val, y_val_pred, zero_division=0))
print("Confusion matrix (val):\n", confusion_matrix(y_val, y_val_pred))

print("\nTest results:")
y_test_pred = model.predict(X_test_scaled)
print(classification_report(y_test, y_test_pred, zero_division=0))
print("Confusion matrix (test):\n", confusion_matrix(y_test, y_test_pred))

# ---- 9. Save artifacts ----
# Save model, scaler, selected_features, target classes, and an example CSV
export_model_path = EXPORT_DIR / "xgboost_model.joblib"
export_scaler_path = EXPORT_DIR / "scaler.joblib"
export_features_path = EXPORT_DIR / "selected_features.json"
export_stage_classes = EXPORT_DIR / "stage_classes.json"
export_example_csv = EXPORT_DIR / "example_inputs.csv"
export_train_for_shap = EXPORT_DIR / "X_train_for_shap.csv"

joblib.dump(model, export_model_path)
joblib.dump(scaler, export_scaler_path)

with open(export_features_path, "w") as f:
    json.dump(feature_cols, f)

with open(export_stage_classes, "w") as f:
    json.dump(list(le_target.classes_), f)

# example inputs (first 10 rows of original features)
pd.DataFrame(X_test.iloc[:10]).to_csv(export_example_csv, index=False)

# Save X_train_scaled (balanced or original) for SHAP if you want:
pd.DataFrame(X_train_scaled, columns=feature_cols).to_csv(export_train_for_shap, index=False)

print("\nArtifacts exported:")
print(export_model_path, export_scaler_path, export_features_path, export_stage_classes, export_example_csv)
