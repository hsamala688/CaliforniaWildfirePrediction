# California Wildfire Prediction

A machine learning project that predicts wildfire risk across California using geospatial and environmental data. Built as a collaborative project for the **NSDC Winter 2026 Presentation**.

## 📌 Overview

California faces growing wildfire threats each year. This project leverages machine learning — including a Random Forest classifier — combined with geographic seed zone data to model and predict wildfire risk across the state. The pipeline covers data ingestion and engineering, model training and evaluation, and an API layer for serving predictions.

## 📁 Project Structure

```
CaliforniaWildfirePrediction/
│
├── DataEngineering.ipynb              # data engineering notebook
├── dataengineering.py                 # Data processing and feature engineering pipeline
├── mlmodels.py                        # ML model training and evaluation
│
├── california.geojson                 # California state boundary geometry
├── California_Seed_Zones_*.geojson    # California seed zone geographic data
│
├── requirements.txt                   # Allows install of required python packages
└── .gitignore
```

## 🧠 Models & Approach

- **Random Forest Classifier** — primary model for predicting wildfire occurrence or risk level
- Geographic seed zones are used as spatial features to capture regional vegetation and climate patterns
- The data engineering pipeline prepares and transforms raw inputs into model-ready features

## 🛠️ Tech Stack

- **Python 3.x**
- **Jupyter Notebook**
- **scikit-learn** — machine learning models
- **pandas / numpy** — data manipulation
- **geopandas / shapely** — geospatial data processing
- **GeoJSON** — geographic boundary and zone data

## 🚀 Getting Started

### 1. Clone the repository

```bash
git clone https://github.com/hsamala688/CaliforniaWildfirePrediction.git
cd CaliforniaWildfirePrediction
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Run the data engineering pipeline

```bash
python dataengineering.py
```

### 4. Train and evaluate the model

```bash
python mlmodels.py
```

## 🗺️ Geographic Data

Two GeoJSON files are included:

- **`california.geojson`** — the state boundary of California for map rendering and spatial filtering
- **`California_Seed_Zones_*.geojson`** — California seed zones, used as regional geographic features in the prediction model

---

## 👥 Contributors

This project was built collaboratively as part of the **NSDC (National Student Data Corps) @ UCLA Winter 2026** project showcase.

---

## 📄 License

This project is open source. Feel free to fork, use, and build upon it.
