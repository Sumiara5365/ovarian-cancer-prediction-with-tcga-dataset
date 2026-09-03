import pickle
import numpy as np

# Load saved artifacts
model = pickle.load(open("xgb_model.pkl", "rb"))
scaler = pickle.load(open("scaler.pkl", "rb"))
le = pickle.load(open("feature_encoders.pkl", "rb"))

# Example input (replace with real values)
sample_input = np.array([[3.4, 7.1, 9.3, 5.6]])  # shape (1, n_features)

# Scale input
sample_scaled = scaler.transform(sample_input)

# Predict
prediction = model.predict(sample_scaled)
predicted_stage = le.inverse_transform(prediction)

print("Predicted Stage:", predicted_stage[0])
