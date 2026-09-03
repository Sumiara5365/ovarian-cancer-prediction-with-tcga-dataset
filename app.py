# app.py
import streamlit as st
import pandas as pd
import numpy as np
import joblib, json
import shap
import matplotlib.pyplot as plt

st.set_page_config(layout="wide", page_title="Ovarian Cancer Stage Predictor")
st.title("Ovarian Cancer Stage Predictor and  Explainer")

@st.cache_data
def load_artifacts():
    model = joblib.load("xgboost_model.joblib")
    scaler = joblib.load("scaler.joblib")
    features = joblib.load("feature_list.joblib")
    feature_encoders = joblib.load("feature_encoders.joblib")
    target_encoder = joblib.load("target_encoder.joblib")

    with open("stage_classes.json", "r") as f:
        stage_classes = json.load(f)

    try:
        examples = pd.read_csv("example_inputs.csv")
    except:
        examples = None

    return model, scaler, features, feature_encoders, target_encoder, stage_classes, examples


model, scaler, features, feature_encoders, target_encoder, stage_classes, examples = load_artifacts()

# -------------------------------------------
# INPUT MODE
# -------------------------------------------
st.sidebar.header("Input")
input_mode = st.sidebar.radio("Input mode", ["Upload CSV", "Choose example (first 10 rows)"])

uploaded = None
chosen_idx = None

if input_mode == "Upload CSV":
    uploaded = st.sidebar.file_uploader("Upload CSV (must include feature columns)", type=["csv"])
else:
    if examples is not None:
        chosen_idx = st.sidebar.selectbox("Choose example row index", examples.index.tolist())
        st.sidebar.write("Example preview:")
        st.sidebar.dataframe(examples)

top_n = st.sidebar.slider("Top biomarkers to display (per class)", 5, 100, 30, 5)

# -------------------------------------------
# PREPROCESSING
# -------------------------------------------
def preprocess_input(df_in, features, feature_encoders, scaler):
    drop_cols = ['CLINICAL_STAGE', 'CLINICAL STAGE', 'Stage', 'stage']
    df_in = df_in.drop(columns=[c for c in drop_cols if c in df_in.columns], errors='ignore')

    missing = [c for c in features if c not in df_in.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing[:10]}{'...' if len(missing)>10 else ''}")

    X = df_in[features].copy()

    for col, le in feature_encoders.items():
        if col in X.columns:
            X[col] = X[col].astype(str)
            X[col] = X[col].apply(lambda v: v if v in le.classes_ else le.classes_[0])
            X[col] = le.transform(X[col])

    X = X.apply(pd.to_numeric, errors='coerce')

    if X.isnull().any().any():
        raise ValueError("Input contains non-numeric or missing values after encoding.")

    X_scaled = pd.DataFrame(scaler.transform(X), columns=features, index=X.index)
    return X_scaled


# --- Load sample ---
if input_mode == "Upload CSV" and uploaded is not None:
    try:
        df_input = pd.read_csv(uploaded)
        X_scaled = preprocess_input(df_input, features, feature_encoders, scaler)
        st.success("Uploaded and preprocessed successfully.")
        sample_indices = X_scaled.index.tolist()
    except Exception as e:
        st.error(f"Error processing input: {e}")
        st.stop()

elif input_mode == "Choose example" and examples is not None:
    df_input = examples.copy()
    try:
        X_scaled = preprocess_input(df_input, features, feature_encoders, scaler)
        st.success("Loaded example dataset.")
        sample_indices = X_scaled.index.tolist()
    except Exception as e:
        st.error(f"Error: {e}")
        st.stop()

else:
    st.info("Please upload a CSV or choose an example.")
    st.stop()


# -------------------------------------------
# SELECT SAMPLE
# -------------------------------------------
sel_index = st.selectbox("Select sample row index to analyze", sample_indices)
X_sample = X_scaled.loc[[sel_index]]
orig_sample = df_input.loc[[sel_index]]

# -------------------------------------------
# PREDICTION
# -------------------------------------------
probs = model.predict_proba(X_sample)
pred = model.predict(X_sample)
pred_label = target_encoder.inverse_transform(pred.astype(int))[0]

st.subheader("Prediction")
col1, col2 = st.columns([1,1])

with col1:
    st.metric("Predicted Stage", pred_label)
    st.write("Probabilities:")
    prob_series = pd.Series(probs[0], index=stage_classes)
    st.table(pd.DataFrame(prob_series, columns=["Probability"]))

with col2:
    st.write("Original feature values (first 10 columns):")
    st.dataframe(orig_sample.iloc[:, :10].T)

# -------------------------------------------
# SHAP FIX
# -------------------------------------------
# ============================
# SHAP explanation (FIXED)
# ============================
# ============================================================
# ---------------------- SHAP SECTION ------------------------
# ============================================================

st.subheader("SHAP Explanations")

# Convert X into numpy for SHAP
X_np = X_scaled.values.astype(np.float32)
n_samples, n_features = X_np.shape

# Build SHAP explainer (using booster prevents bugs)
try:
    booster = model.get_booster()
    explainer = shap.TreeExplainer(booster, feature_perturbation="interventional")
except:
    explainer = shap.TreeExplainer(model, feature_perturbation="interventional")

raw_shap = explainer.shap_values(X_np)

# ---------------------- SHAP EXPLANATIONS ----------------------------
st.subheader("SHAP Explanations")

# Use correct SHAP input format
X_for_shap = X_scaled.values.astype(np.float32)
n_samples, n_features = X_for_shap.shape

# Build TreeExplainer
try:
    explainer = shap.TreeExplainer(model.get_booster())
except:
    explainer = shap.TreeExplainer(model)

raw_shap = explainer.shap_values(X_for_shap)

# ---- Normalize all possible SHAP output formats ----
def normalize_shap_output(shap_out, n_features):
    """
    Normalizes SHAP output for:
      - list of (samples, features)
      - list of (samples, features, 1)
      - array: (samples, features)
      - array: (samples, features, classes)
      - array: (classes, samples, features)
    """
    # Case 1: XGBoost multiclass often returns list of arrays
    if isinstance(shap_out, list):
        fixed = []
        for arr in shap_out:
            arr = np.asarray(arr)
            # Remove trailing dim 1
            if arr.ndim == 3 and arr.shape[-1] == 1:
                arr = arr[:, :, 0]
            # Final shape must be (samples, features)
            if arr.ndim != 2 or arr.shape[1] != n_features:
                raise ValueError(f"SHAP shape mismatch inside list: {arr.shape}")
            fixed.append(arr)
        return fixed

    # Case 2: SHAP returns single ndarray
    arr = np.asarray(shap_out)

    # Shapes that must be fixed:
    # (samples, features, classes) → split into list
    if arr.ndim == 3 and arr.shape[-1] > 1:
        # Example: (302, 102, 4)
        if arr.shape[1] != n_features:
            raise ValueError(f"SHAP mismatch: expected {n_features} features, got {arr.shape[1]}")
        # Produce list of shape: list[ (samples, features) ]
        out = [arr[:, :, c] for c in range(arr.shape[-1])]
        return out

    # (classes, samples, features)
    if arr.ndim == 3 and arr.shape[0] > 1:
        if arr.shape[2] != n_features:
            raise ValueError(f"SHAP mismatch: expected {n_features} features, got {arr.shape[2]}")
        out = [arr[c, :, :] for c in range(arr.shape[0])]
        return out

    # Squeeze leftover extra dim
    arr = np.squeeze(arr)

    # After squeeze → must be (samples, features)
    if arr.ndim != 2 or arr.shape[1] != n_features:
        raise ValueError(f"SHAP shape incorrect after squeeze: {arr.shape}")

    return arr


# --- Apply normalization ---
shap_values = normalize_shap_output(raw_shap, n_features)

# ---------------- GLOBAL IMPORTANCE ----------------
if isinstance(shap_values, list):
    # Shape: list[ (samples, features) ]
    shap_abs = np.stack([np.abs(sv) for sv in shap_values])   # (classes, samples, features)
    mean_shap_per_feature = shap_abs.mean(axis=(0, 1))        # → (features,)
else:
    mean_shap_per_feature = np.abs(shap_values).mean(axis=0)

# Build dataframe
feat_imp = pd.DataFrame({
    "feature": features,
    "mean_abs_shap": mean_shap_per_feature
}).sort_values("mean_abs_shap", ascending=False)

st.markdown("### Global feature importance (mean |SHAP|)")
fig, ax = plt.subplots(figsize=(8, min(12, 0.15 * len(feat_imp))))
ax.barh(feat_imp["feature"].head(100)[::-1], feat_imp["mean_abs_shap"].head(100)[::-1])
plt.tight_layout()
st.pyplot(fig)

# ---------------- PER SAMPLE ANALYSIS ----------------
st.markdown("### Top contributing biomarkers for selected sample")

sample_pos = list(X_scaled.index).index(sel_index)

if isinstance(shap_values, list):
    # Multiclass
    for i, cls in enumerate(stage_classes):
        sv_sample = shap_values[i][sample_pos]
        df_s = pd.DataFrame({"feature": features, "shap_value": sv_sample})
        df_s["abs"] = df_s["shap_value"].abs()
        df_s = df_s.sort_values("abs", ascending=False).head(top_n)

        st.markdown(f"#### Class: {cls}")
        fig, ax = plt.subplots(figsize=(8, 0.18 * len(df_s)))
        ax.barh(df_s["feature"][::-1], df_s["abs"][::-1])
        plt.tight_layout()
        st.pyplot(fig)
        st.dataframe(df_s[["feature", "shap_value"]].reset_index(drop=True))
else:
    # Single output model
    sv_sample = shap_values[sample_pos]
    df_s = pd.DataFrame({"feature": features, "shap_value": sv_sample})
    df_s["abs"] = df_s["shap_value"].abs()
    df_s = df_s.sort_values("abs", ascending=False).head(top_n)

    fig, ax = plt.subplots(figsize=(8, 0.18 * len(df_s)))
    ax.barh(df_s["feature"][::-1], df_s["abs"][::-1])
    plt.tight_layout()
    st.pyplot(fig)
    st.dataframe(df_s)
