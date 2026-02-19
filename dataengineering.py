!pip install "meteostat<2.0"
import pandas as pd
import numpy as np
from meteostat import Stations, Daily, Point #this is used to help you guys pull all the data from the relevant stations within California


# This script is used to clean the Archive FIRMS Data

# -----------------------------
# CONFIG
# -----------------------------
INPUT_CSV = "fire_archive_SV-C2_710372.csv"
OUTPUT_CSV = "fire_archive_cleaned.csv"

# -----------------------------
# LOAD DATA
# -----------------------------
df = pd.read_csv(INPUT_CSV)

# -----------------------------
# STANDARDIZE COLUMN NAMES
# -----------------------------
df.columns = df.columns.str.lower().str.strip()

# -----------------------------
# DROP DUPLICATES
# -----------------------------
df = df.drop_duplicates()

# -----------------------------
# QUALITY FILTERS
# -----------------------------
# Keep only nominal & high-confidence detections
if "confidence" in df.columns:
    df = df[df["confidence"].isin(["n", "h"])]

# Remove obviously bad FRP values
if "frp" in df.columns:
    df = df[df["frp"] > 0]

# Remove bad brightness values
if "bright_ti4" in df.columns:
    df = df[df["bright_ti4"] > 0]

# -----------------------------
# DATE / TIME HANDLING
# -----------------------------
# Convert acq_date to datetime
df["acq_date"] = pd.to_datetime(df["acq_date"], errors="coerce")

# Convert acq_time to HH:MM format
df["acq_time"] = df["acq_time"].astype(str).str.zfill(4)
df["acq_hour"] = df["acq_time"].str[:2].astype(int)
df["acq_minute"] = df["acq_time"].str[2:].astype(int)

# Combine into a single timestamp
df["timestamp_utc"] = (
    df["acq_date"]
    + pd.to_timedelta(df["acq_hour"], unit="h")
    + pd.to_timedelta(df["acq_minute"], unit="m")
)

# -----------------------------
# GEOGRAPHIC SANITY CHECKS
# -----------------------------
df = df[
    (df["latitude"].between(-90, 90)) &
    (df["longitude"].between(-180, 180))
]

# -----------------------------
# DROP UNUSED / REDUNDANT COLUMNS
# -----------------------------
columns_to_drop = [
    "acq_time",        # replaced by timestamp
    "bright_ti5",      # often redundant
    "version",         # metadata
    "confidence"       # already filtered
]

df = df.drop(columns=[c for c in columns_to_drop if c in df.columns])

# -----------------------------
# SORT + RESET INDEX
# -----------------------------
df = df.sort_values("timestamp_utc").reset_index(drop=True)

# -----------------------------
# SAVE CLEAN FILE
# -----------------------------
df.to_csv(OUTPUT_CSV, index=False)


# Because the NRT data is more sensitive to clean, this cell is dedicated
# to the NRT dataset.

# -----------------------------
# CONFIG
# -----------------------------
INPUT_CSV = "fire_nrt_SV-C2_710372.csv"
OUTPUT_CSV = "fire_nrt_cleaned.csv"

# -----------------------------
# LOAD DATA
# -----------------------------
df = pd.read_csv(INPUT_CSV)
df.columns = df.columns.str.lower().str.strip()

# -----------------------------
# DROP DUPLICATES
# -----------------------------
df = df.drop_duplicates()

# -----------------------------
# BASIC SANITY CHECKS (NOT AGGRESSIVE)
# -----------------------------
df = df[
    (df["latitude"].between(-90, 90)) &
    (df["longitude"].between(-180, 180))
]

# -----------------------------
# FRP / BRIGHTNESS SANITY
# (flag, don't drop)
# -----------------------------
df["frp_valid"] = True
df.loc[df["frp"] <= 0, "frp_valid"] = False

if "bright_ti4" in df.columns:
    df["bright_valid"] = True
    df.loc[df["bright_ti4"] <= 0, "bright_valid"] = False

# -----------------------------
# CONFIDENCE HANDLING
# -----------------------------
# Convert confidence to numeric scale
confidence_map = {
    "l": 0,
    "n": 1,
    "h": 2
}

if "confidence" in df.columns:
    df["confidence_level"] = df["confidence"].map(confidence_map)
else:
    df["confidence_level"] = None

# -----------------------------
# DATE / TIME HANDLING
# -----------------------------
df["acq_date"] = pd.to_datetime(df["acq_date"], errors="coerce")

df["acq_time"] = df["acq_time"].astype(str).str.zfill(4)
df["acq_hour"] = df["acq_time"].str[:2].astype(int)
df["acq_minute"] = df["acq_time"].str[2:].astype(int)

df["timestamp_utc"] = (
    df["acq_date"]
    + pd.to_timedelta(df["acq_hour"], unit="h")
    + pd.to_timedelta(df["acq_minute"], unit="m")
)

# -----------------------------
# SATELLITE / INSTRUMENT FLAGS
# -----------------------------
if "instrument" in df.columns:
    df["is_viirs"] = df["instrument"].str.contains("VIIRS", na=False)

# -----------------------------
# DROP ONLY TRUE REDUNDANCY
# -----------------------------
columns_to_drop = [
    "acq_time",   # replaced by timestamp
]

df = df.drop(columns=[c for c in columns_to_drop if c in df.columns])

# -----------------------------
# SORT + SAVE
# -----------------------------
df = df.sort_values("timestamp_utc").reset_index(drop=True)
df.to_csv(OUTPUT_CSV, index=False)



# Cross-checking FIRMS Archive Data with 5 most recent fires from CAL FIRE
''' (I didn't know how to convert the data from CAL FIRE into a CSV file,
so I wrote it by hand. I added it to the Raw Data folder on the Drive.
The title is "cal_fire_top_5.csv")
-Arjun '''

# Import geodesic for easier distance calculation
from geopy.distance import geodesic

# -----------------------------
# CONFIG
# -----------------------------
FIRMS_CSV = "fire_archive_cleaned.csv"
CALFIRE_CSV = "cal_fire_top_5.csv"
OUTPUT_CSV = "firms_calfire_cross_reference.csv"

MAX_DISTANCE_KM = 10      # spatial tolerance
TIME_WINDOW_DAYS = 1      # ± days around start date

# -----------------------------
# LOAD DATA
# -----------------------------
firms = pd.read_csv(FIRMS_CSV, parse_dates=["timestamp_utc"])
calfire = pd.read_csv(CALFIRE_CSV, parse_dates=["start_date"])

results = []

# -----------------------------
# CROSS-REFERENCE
# -----------------------------
for _, fire in calfire.iterrows():
    fire_point = (fire["latitude"], fire["longitude"])

    # time window
    start = fire["start_date"] - pd.Timedelta(days=TIME_WINDOW_DAYS)
    end = fire["start_date"] + pd.Timedelta(days=TIME_WINDOW_DAYS)

    # FIRMS detections in time window
    firms_subset = firms[
        (firms["timestamp_utc"] >= start) &
        (firms["timestamp_utc"] <= end)
    ]

    for _, f in firms_subset.iterrows():
        firms_point = (f["latitude"], f["longitude"])
        distance_km = geodesic(fire_point, firms_point).km

        if distance_km <= MAX_DISTANCE_KM:
            results.append({
                "incident_name": fire["incident_name"],
                "incident_lat": fire["latitude"],
                "incident_lon": fire["longitude"],
                "firms_lat": f["latitude"],
                "firms_lon": f["longitude"],
                "distance_km": round(distance_km, 2),
                "firms_time": f["timestamp_utc"]
            })

# -----------------------------
# SAVE RESULTS
# -----------------------------
matches_df = pd.DataFrame(results)
matches_df.to_csv(OUTPUT_CSV, index=False)



# This script adds the seed zones into the cleaned up FIRMS Archive File
''' I got a bit of help for this because I'm not the best with geopandas.
If anything seems off after running this script, please let me know and
I'll fix it asap.
-Arjun '''

# Importing geopandas library due to geojson file of seed zones
import geopandas as gpd

# -----------------------------
# LOAD DATA
# -----------------------------
firms = pd.read_csv("fire_archive_cleaned.csv")

# Convert FIRMS to GeoDataFrame
firms_gdf = gpd.GeoDataFrame(
    firms,
    geometry=gpd.points_from_xy(firms.longitude, firms.latitude),
    crs="EPSG:4326"
)

# Load seed zone polygons
seed_zones = gpd.read_file("California_Seed_Zones_3280520806235389701.geojson")

# Ensure same CRS
seed_zones = seed_zones.to_crs(firms_gdf.crs)

# -----------------------------
# SPATIAL JOIN
# -----------------------------
firms_with_seed = gpd.sjoin(
    firms_gdf,
    seed_zones[["SEED_ZONE", "REGION", "SUBREGION", "SUBZONE", "geometry"]],
    how="left",
    predicate="within"
)

# Drop geometry for CSV output
firms_with_seed = firms_with_seed.drop(columns="geometry")

firms_with_seed.to_csv("fire_firms_with_seed_zones.csv", index=False)

# If you want to locally download the finished file, delete the hashtag below:
# files.download("fire_firms_with_seed_zones.csv")


from shapely.geometry import Point
from shapely.ops import unary_union
from pyproj import Transformer
import random

fires = pd.read_csv("fire_firms_with_seed_zones.csv")
fires["fire"] = 1
fires["acq_date"] = pd.to_datetime(fires["acq_date"])

ca = gpd.read_file("california_boundary.shp")
ca = ca.to_crs(epsg=4326)

ca_polygon = unary_union(ca.geometry)

def random_point():
    minx, miny, maxx, maxy = ca_polygon.bounds
    while True:
        p = Point(
            random.uniform(minx, maxx),
            random.uniform(miny, maxy)
        )
        if ca_polygon.contains(p):
            return p
        
transformer = Transformer.from_crs("EPSG:4326", "EPSG:5070", always_xy = True)

def to_albers(lon, lat):
    return transformer.transform(lon, lat)

fires["x"], fires["y"] = zip(*fires.apply(
    lambda r: to_albers(r["longitude"], r["latitude"]), axis = 1
))

n = int(len(fires) * 0.5)
no_fire_rows = []

fires_by_date = fires.groupby("acq_date")

while len(no_fire_rows) < n:
    date = random.choice(fires["acq_date"].unique())
    day_fire = fires_by_date.get_group(date)

    p = random_point()
    lon, lat = p.x, p.y
    x, y = to_albers(lon, lat)

    dists = np.sqrt((day_fire["x"] - x)**2 + (day_fire["y"] - y)**2)

    if dists.min() > 5_000:
        row = {
            "latitude": lat,
            "longitude": lon,
            "acq_date": date,
            "fire": 0
        }
        no_fire_rows.append(row)

no_fire = pd.DataFrame(no_fire_rows)

for col in fire.columns:
    if col not in no_fire.columns:
        no_fire[col] = np.nan

master = pd.concat([fire, no_fire], ignore_index=True)


''' Got my data from LANDFIRE, extracted .tif of CONUS for Existing Vegetation Type, Height, Cover (denoted evt, evh, evc respectively) and filtered it to only contain data from california (all done on a separate doc since I couldn't just download California data off rip). Code below is mostly done with the help of chat since I've never played with rasters and .tif fiels 
-Will '''

import rasterio

df = pd.read_csv("fire_firms_with_seed_zones.csv")

gdf = gpd.GeoDataFrame(
    df,
    geometry=gpd.points_from_xy(df["longitude"], df["latitude"]), crs="EPSG:4326"
)

def add_raster_values(gdf, raster_path, out_col):
    with rasterio.open(raster_path) as src:
        pts = gdf.to_crs(src.crs)

        coords = [(geom.x, geom.y) for geom in pts.geometry]
        vals = [v[0] for v in src.sample(coords)]

        nodata = src.nodata
        if nodata is not None:
            vals = [None if (v == nodata) else v for v in vals]

        gdf[out_col] = vals
    return gdf

gdf = add_raster_values(gdf, "evt_ca.tif", "lf_evt")
gdf = add_raster_values(gdf, "evc_ca.tif", "lf_evc")
gdf = add_raster_values(gdf, "evh_ca.tif", "lf_evh")



'''Merged lookup table adding columns 'lf_evc', 'lf_evh',
       'EVT_NAME', 'LFRDB', 'EVT_FUEL', 'EVT_FUEL_N', 'EVT_LF', 'EVT_PHYS',
       'EVT_GP', 'EVT_GP_N', 'SAF_SRM', 'EVT_ORDER', 'EVT_CLASS', 'EVT_SBCLS' to provide context on vegetation values 
-Will '''

gdf = gdf.dropna(subset=["lf_evt", "lf_evc", "lf_evh"])
evt_lookup = pd.read_csv("LF2024_EVT.csv")
evt_lookup["VALUE"] = evt_lookup["VALUE"].astype(float)
evt_lookup = evt_lookup.drop(columns=["R", "G", "B", "RED", "GREEN", "BLUE"])
gdf = gdf.merge(
    evt_lookup,
    left_on = "lf_evt",
    right_on = "VALUE",
    how="left"
)
gdf = gdf.drop(columns=["lf_evt", "VALUE", "LFRDB", "EVT_ORDER", "SAF_SRM", "geometry"])





