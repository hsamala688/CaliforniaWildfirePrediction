import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline

# -----------------------
# make dummy data
# -----------------------
np.random.seed(42)
n = 500

df = pd.DataFrame({
    "tmmx": np.random.normal(30, 6, n), # max temp
    "tmmn": np.random.normal(15, 5, n), # min temp
    "pr": np.random.exponential(3, n), # precipitation
    "vs": np.random.normal(4, 2, n), # wind speed
    "vpd": np.random.normal(2, 1, n), #vapor pressure deficit (air dryness)
    "erc": np.random.normal(60, 20, n), # energy release component
    "bi": np.random.normal(80, 25, n), # burning index
    "fm100": np.random.normal(18, 5, n), # moisture in small fuel (small twigs, leaves, etc.)
    "fm1000": np.random.normal(25, 6, n), # moisture in large fuels (big logs, etc.)
})

# wildfire rule: hot + dry + windy
# less moisture
logit = (
    -5
    +0.08 * df["tmmx"] 
    +0.6 * df["vpd"]
    +0.1 * df["vs"]
    -0.05 * df["fm100"]
    )

p_fire = 1 / (1 + np.exp(-logit))
df["fire"] = np.random.binomial(1, np.clip(p_fire, 0, 0.4))

print("Fire count: ", df["fire"].value_counts())

# -----------------------
# model
# -----------------------
X = df.drop(columns = "fire")
y = df["fire"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size = 0.25,
    random_state = 42,
    stratify=y
)

model = Pipeline([
    ("scaler", StandardScaler()),
    ("rf", RandomForestClassifier(
        n_estimators=200,
        random_state=42,
        class_weight="balanced"
    ))
])

model.fit(X_train, y_train)

# y_pred = model.predict(X_test) # predicts fire if prob > 0.5
y_prob = model.predict_proba(X_test)[:,1]
y_pred = (y_prob > 0.2).astype(int) # predicts fire if prob > 0.2

print("\nClassification Report")
print(classification_report(y_test, y_pred, digits=3))

print("Confusion Matrix")
print(confusion_matrix(y_test, y_pred))

# -----------------------
# feature importance
# -----------------------
rf = model.named_steps["rf"]
importances = rf.feature_importances_

plt.figure(figsize=(7,5))
plt.barh(X.columns, importances)
plt.title("Feature Importance")
plt.xlabel("Importance")
plt.tight_layout()
plt.show()

# -----------------------
# confusion matrix heat map fin later
# -----------------------