import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)
from sklearn.model_selection import train_test_split

from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

# -----------------------
# import the master csv
# -----------------------
file_path = "master_final.csv"

df = pd.read_csv(file_path)
df = df.drop(
    columns=[
        "satellite",
        "instrument",
        "daynight",
        "timestamp_utc",
        "nearest_station_id",
        "brightness",
        "frp",
        "bright_t31",
        "type",
    ],
    errors="ignore",
)

df = pd.get_dummies(df, columns=["EVT_FUEL_N"], drop_first=True)

df["date"] = pd.to_datetime(df["date"])
df["month"] = df["date"].dt.month
df["day_of_year"] = df["date"].dt.dayofyear
df["year"] = df["date"].dt.year

df = df.drop(columns=["date"])

X = df[
    [
        "latitude",
        "longitude",
        "wx_tavg_c",
        "wx_prcp_mm",
        "wx_wspd_ms",
        "snow",
        "lf_evc",
        "lf_evh",
    ]
    + [col for col in df.columns if col.startswith("EVT_FUEL_N_")]
]

y = df["fire"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42, stratify=y
)

model = RandomForestClassifier(
    n_estimators=100,  # Reduced from 200
    max_depth=10,  # CRITICAL: This keeps the file size small
    min_samples_leaf=50,
    random_state=42,
    class_weight="balanced",  # important if fire is rare
    n_jobs=-1,
)

model.fit(X_train, y_train)

y_pred = model.predict(X_test)  # predicts fire if prob > 0.5
y_prob = model.predict_proba(X_test)[:, 1]

# -----------------------
# regression type evaluation
# -----------------------

r2 = r2_score(y_test, y_prob)
mae = mean_absolute_error(y_test, y_prob)
rmse = np.sqrt(mean_squared_error(y_test, y_prob))

print(f"\nR^2: {r2:.4f}")
print(f"Mean Absolute Error (MAE): {mae:.4f}")
print(f"Root Mean Squared Error (RMSE): {rmse:.4f}")

# Evaluation

accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy * 100:.2f}%")

conf_matrix = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(6, 5))
sns.heatmap(
    conf_matrix,
    annot=True,
    fmt="g",
    cmap="Reds",
    cbar=False,
    xticklabels=["No Fire", "Fire"],
    yticklabels=["No Fire", "Fire"],
)

plt.title("Confusion Matrix - Fire Prediction")
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.show()

print("\nClassification Report")
print(classification_report(y_test, y_pred, target_names=["No Fire", "Fire"]))

print("Confusion Matrix")
print(confusion_matrix(y_test, y_pred))

# -----------------------
# feature importance
# -----------------------

rf = model
importances = rf.feature_importances_
feature_names = np.array(X.columns)

# Sort features by importance
indices = np.argsort(importances)[::-1]

# Select top 20
top_n = 20
top_features = feature_names[indices][:top_n]
top_importances = importances[indices][:top_n]

# -----------------------
# feature names clean
# -----------------------
clean_names = {
    "day_of_year": "Day of Year",
    "month": "Month",
    "year": "Year",
    "latitude": "Latitude",
    "longitude": "Longitude",
    "wx_tavg_c": "Avg Temperature (°C)",
    "wx_prcp_mm": "Precipitation (mm)",
    "wx_wspd_ms": "Wind Speed (m/s)",
    "lf_evc": "Vegetation Cover",
    "lf_evh": "Vegetation Height",
}

<<<<<<< HEAD

=======
>>>>>>> cc9123d189a6b6a9ca40ab18784fc2e3122ef4f7
def clean_feature_name(name):
    name = clean_names.get(name, name)

    name = name.replace("EVT_FUEL_N_", "")

    name = name.replace("Mediterranean California", "Med. CA")
    name = name.replace("North American", "N. American")
    name = name.replace("Sparsely Vegetated Systems", "Sparse Veg.")
    name = name.replace("Mixed Conifer Forest and Woodland", "Mixed Conifer")

    name = name.replace("_", " ")

    return name

<<<<<<< HEAD

=======
>>>>>>> cc9123d189a6b6a9ca40ab18784fc2e3122ef4f7
top_features_clean = [clean_feature_name(f) for f in top_features]

# -----------------------
# Plot
# -----------------------
plt.figure(figsize=(8, 6))

plt.style.use("seaborn-v0_8-whitegrid")
colors = plt.cm.Reds(np.linspace(0.3, 0.9, len(top_importances)))

plt.barh(top_features_clean[::-1], top_importances[::-1], color=colors)
plt.title("Top 20 Feature Importances")
plt.xlabel("Importance")

plt.grid(axis="x", linestyle="--", alpha=0.5)
plt.grid(axis="y", visible=False)

plt.gca().spines["top"].set_visible(False)
plt.gca().spines["right"].set_visible(False)

plt.tight_layout()
plt.show()

joblib.dump(model, "wildfire_model.pkl")
fuel_cols = [col for col in df.columns if col.startswith("EVT_FUEL_N_")]
joblib.dump(fuel_cols, "fuel_encoder.pkl")
joblib.dump(X.columns.tolist(), "feature_names.pkl")
