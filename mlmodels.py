import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split

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
        "month",
        "day_of_year",
        "year",
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
# print(np.percentile(y_prob, [25, 50, 70, 85, 95, 99]))
# Evaluation

accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy * 100:.2f}%")

conf_matrix = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(6, 5))
sns.heatmap(
    conf_matrix,
    annot=True,
    fmt="g",
    cmap="Blues",
    cbar=False,
    xticklabels=["No Fire", "Fire"],
    yticklabels=["No Fire", "Fire"],
)

plt.title("Confusion Matrix - Fire Prediction")
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.show()
plt.close("all")

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

# Plot
plt.figure(figsize=(8, 6))
plt.barh(top_features[::-1], top_importances[::-1])
plt.title("Top 20 Feature Importances")
plt.xlabel("Importance")
plt.tight_layout()
plt.show()

joblib.dump(model, "wildfire_model.pkl", compress=3)
fuel_cols = [col for col in df.columns if col.startswith("EVT_FUEL_N_")]
joblib.dump(fuel_cols, "fuel_encoder.pkl")
joblib.dump(X.columns.tolist(), "feature_names.pkl")
