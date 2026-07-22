import pandas as pd
import joblib

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error

# -----------------------------
# Load Training Dataset
# -----------------------------
df = pd.read_csv("training_dataset.csv")

# -----------------------------
# Select Features
# -----------------------------
X = df[
    [
        "mfcc_mean",
        "zcr",
        "centroid",
        "bandwidth",
        "rms"
    ]
]

# Target
y = df["bird_count"]

# -----------------------------
# Split Dataset
# -----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# -----------------------------
# Create Model
# -----------------------------
model = RandomForestRegressor(
    n_estimators=100,
    random_state=42
)

# -----------------------------
# Train Model
# -----------------------------
model.fit(X_train, y_train)

# -----------------------------
# Predict
# -----------------------------
predictions = model.predict(X_test)

# -----------------------------
# Evaluate
# -----------------------------
mae = mean_absolute_error(
    y_test,
    predictions
)

print(f"Mean Absolute Error: {mae:.2f}")

# -----------------------------
# Save Model
# -----------------------------
joblib.dump(
    model,
    "bird_model.pkl"
)

print("Model saved as bird_model.pkl")