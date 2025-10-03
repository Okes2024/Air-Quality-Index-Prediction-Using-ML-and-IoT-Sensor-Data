#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Air-Quality-Index-Prediction-Using-ML-and-IoT-Sensor-Data (Synthetic)
---------------------------------------------------------------------
This script simulates an IoT air-quality network, generates a reasonably
populated dataset with timestamps and locations, computes AQI from PM2.5/PM10
(US EPA breakpoints), trains ML models to predict AQI, evaluates them, and saves:
  - synthetic_air_quality_iot.csv
  - synthetic_air_quality_iot.xlsx
  - model_feature_importance.png
  - residuals_plot.png
  - trained_aqi_model.joblib

Usage:
  python air_quality_iot_ml.py
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import Ridge
from joblib import dump

RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

# -----------------------------
# 1) Generate synthetic IoT air quality dataset
# -----------------------------
n_sensors = 12
days = 14
freq_minutes = 10  # data every 10 minutes
points_per_day = int(24*60/freq_minutes)
n_points = points_per_day * days * n_sensors  # reasonably populated

# Spatial extent (approx around Niger Delta / Southern Nigeria)
lat_min, lat_max = 4.8, 6.0
lon_min, lon_max = 5.0, 7.0

sensor_ids = [f"S{str(i).zfill(3)}" for i in range(1, n_sensors+1)]
sensor_lat = np.random.uniform(lat_min, lat_max, n_sensors)
sensor_lon = np.random.uniform(lon_min, lon_max, n_sensors)

start_time = datetime(2025, 1, 1, 0, 0, 0)
timestamps = []
sensor_idx = []
for s in range(n_sensors):
    for d in range(days):
        for t in range(points_per_day):
            timestamps.append(start_time + timedelta(days=d, minutes=freq_minutes*t))
            sensor_idx.append(s)

timestamps = np.array(timestamps)
sensor_idx = np.array(sensor_idx)

lats = sensor_lat[sensor_idx]
lons = sensor_lon[sensor_idx]
sensor_ids_rep = np.array(sensor_ids)[sensor_idx]

minute_of_day = np.array([ts.hour*60 + ts.minute for ts in timestamps])
day_of_week = np.array([ts.weekday() for ts in timestamps])

temp = 28 + 4*np.sin(2*np.pi*(minute_of_day/1440)) + np.random.normal(0, 1.2, n_points) - 0.5*(lats - lats.mean())
rh = 75 - 8*np.sin(2*np.pi*(minute_of_day/1440) + np.pi/3) + np.random.normal(0, 3, n_points)
rh = np.clip(rh, 25, 100)
wind = np.abs(np.random.normal(2.5 + 0.3*np.sin(2*np.pi*(day_of_week/7)), 0.8, n_points))
pressure = 1012 + np.random.normal(0, 3, n_points) - 0.2*(lats - lats.mean())

rush = ((minute_of_day%1440 >= 7*60) & (minute_of_day%1440 <= 9*60)) | \
       ((minute_of_day%1440 >= 17*60) & (minute_of_day%1440 <= 20*60))
traffic = 0.3 + 0.5*rush.astype(float) + 0.1*(day_of_week<5).astype(float) + np.random.normal(0, 0.05, n_points)
traffic = np.clip(traffic, 0, 1)

industry_factor = np.random.uniform(0.7, 1.3, n_sensors)
industry = industry_factor[sensor_idx]

pm25_base = 15 + 10*traffic + 0.08*(100 - rh) + 1.2*industry + np.random.normal(0, 5, n_points)
pm10_base = 25 + 12*traffic + 0.05*(100 - rh) + 1.0*industry + np.random.normal(0, 6, n_points)
no2 = 12 + 30*traffic + 0.5*np.maximum(0, 30 - wind*10) + np.random.normal(0, 4, n_points)
so2 = 4 + 10*industry + 0.2*traffic + np.random.normal(0, 1.5, n_points)
co = 0.4 + 0.8*traffic + 0.05*industry + np.random.normal(0, 0.1, n_points)  # mg/m³
o3 = 10 + 0.06*(temp-25)**2 - 0.08*no2 + np.random.normal(0, 3, n_points)   # photochemical

pm25 = np.clip(pm25_base, 1, 300)
pm10 = np.clip(pm10_base, 5, 400)
no2 = np.clip(no2, 1, 200)
so2 = np.clip(so2, 1, 120)
co = np.clip(co, 0.05, 10)
o3 = np.clip(o3, 1, 240)

# AQI calculation
AQI_BP = {
    'PM2.5': [
        (0.0, 12.0, 0, 50),
        (12.1, 35.4, 51, 100),
        (35.5, 55.4, 101, 150),
        (55.5, 150.4, 151, 200),
        (150.5, 250.4, 201, 300),
        (250.5, 350.4, 301, 400),
        (350.5, 500.4, 401, 500)
    ],
    'PM10': [
        (0, 54, 0, 50),
        (55, 154, 51, 100),
        (155, 254, 101, 150),
        (255, 354, 151, 200),
        (355, 424, 201, 300),
        (425, 504, 301, 400),
        (505, 604, 401, 500)
    ]
}

def sub_index(Cp, species):
    for (Clow, Chigh, Ilow, Ihigh) in AQI_BP[species]:
        if Clow <= Cp <= Chigh:
            return (Ihigh - Ilow)/(Chigh - Clow) * (Cp - Clow) + Ilow
    if Cp < AQI_BP[species][0][0]:
        return AQI_BP[species][0][2]
    return AQI_BP[species][-1][3]

aqi_pm25 = np.array([sub_index(c, 'PM2.5') for c in pm25])
aqi_pm10 = np.array([sub_index(c, 'PM10') for c in pm10])
aqi = np.maximum(aqi_pm25, aqi_pm10)

def aqi_category(val):
    if val <= 50: return 'Good'
    if val <= 100: return 'Moderate'
    if val <= 150: return 'USG'
    if val <= 200: return 'Unhealthy'
    if val <= 300: return 'Very Unhealthy'
    return 'Hazardous'

aqi_cat = np.array([aqi_category(v) for v in aqi])

df = pd.DataFrame({
    'timestamp': timestamps,
    'sensor_id': sensor_ids_rep,
    'lat': lats,
    'lon': lons,
    'temp_c': temp,
    'humidity_pct': rh,
    'wind_mps': wind,
    'pressure_hpa': pressure,
    'traffic_idx': traffic,
    'pm25_ugm3': pm25,
    'pm10_ugm3': pm10,
    'no2_ugm3': no2,
    'so2_ugm3': so2,
    'co_mgm3': co,
    'o3_ugm3': o3,
    'aqi': aqi,
    'aqi_cat': aqi_cat
})

df.to_csv('synthetic_air_quality_iot.csv', index=False)
try:
    import openpyxl
    df.to_excel('synthetic_air_quality_iot.xlsx', index=False)
except Exception as e:
    print("Excel export skipped (openpyxl not available):", e)

# Time features
df['hour'] = pd.to_datetime(df['timestamp']).dt.hour
df['dow'] = pd.to_datetime(df['timestamp']).dt.dayofweek

target = 'aqi'
features = ['sensor_id','lat','lon','temp_c','humidity_pct','wind_mps','pressure_hpa',
            'traffic_idx','pm25_ugm3','pm10_ugm3','no2_ugm3','so2_ugm3','co_mgm3','o3_ugm3',
            'hour','dow']

X = df[features]
y = df[target]

num_cols = ['lat','lon','temp_c','humidity_pct','wind_mps','pressure_hpa',
            'traffic_idx','pm25_ugm3','pm10_ugm3','no2_ugm3','so2_ugm3','co_mgm3','o3_ugm3',
            'hour','dow']
cat_cols = ['sensor_id']

preprocess = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), num_cols),
        ('cat', OneHotEncoder(handle_unknown='ignore'), cat_cols)
    ]
)

models = {
    'Ridge': Ridge(random_state=RANDOM_STATE),
    'RandomForest': RandomForestRegressor(n_estimators=250, max_depth=None, min_samples_leaf=2,
                                         n_jobs=-1, random_state=RANDOM_STATE),
    'GradientBoosting': GradientBoostingRegressor(random_state=RANDOM_STATE)
}

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=RANDOM_STATE
)

cv = KFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
cv_scores = {}
pipes = {}

for name, model in models.items():
    pipe = Pipeline(steps=[('prep', preprocess), ('clf', model)])
    pipes[name] = pipe
    scores = cross_val_score(pipe, X_train, y_train, cv=cv, scoring='neg_root_mean_squared_error', n_jobs=-1)
    cv_scores[name] = np.mean(scores)

best_name = max(cv_scores, key=cv_scores.get)
best_pipe = pipes[best_name]
best_pipe.fit(X_train, y_train)

y_pred = best_pipe.predict(X_test)
rmse = mean_squared_error(y_test, y_pred, squared=False)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("Cross-validated mean (-RMSE) per model:")
for k, v in cv_scores.items():
    print(f"  {k}: {v:.3f}")
print("\nBest model:", best_name)
print(f"Test RMSE: {rmse:.3f}")
print(f"Test MAE : {mae:.3f}")
print(f"Test R^2 : {r2:.3f}")

# Residuals plot
residuals = y_test - y_pred
plt.figure(figsize=(6,5))
plt.scatter(y_pred, residuals, s=6)
plt.axhline(0, linestyle='--')
plt.xlabel('Predicted AQI')
plt.ylabel('Residual (Actual - Predicted)')
plt.title('Residuals vs Predicted AQI')
plt.tight_layout()
plt.savefig('residuals_plot.png', dpi=300)
plt.close()

# Feature importance (if available)
fitted = best_pipe.named_steps['clf']
fi_path = None
if hasattr(fitted, 'feature_importances_'):
    ohe = best_pipe.named_steps['prep'].named_transformers_['cat']
    num_features = num_cols
    cat_features = list(ohe.get_feature_names_out(cat_cols))
    all_features = num_features + cat_features

    importances = fitted.feature_importances_
    n_min = min(len(importances), len(all_features))
    order = np.argsort(importances[:n_min])[::-1][:20]
    top_feats = [all_features[i] for i in order]
    top_vals = importances[order]

    plt.figure(figsize=(8,6))
    plt.barh(top_feats[::-1], top_vals[::-1])
    plt.xlabel('Importance')
    plt.title(f'Feature Importance ({best_name})')
    plt.tight_layout()
    fi_path = 'model_feature_importance.png'
    plt.savefig(fi_path, dpi=300)
    plt.close()

from joblib import dump
dump(best_pipe, 'trained_aqi_model.joblib')

print("\nSaved files:")
print("  synthetic_air_quality_iot.csv")
print("  synthetic_air_quality_iot.xlsx")
print("  residuals_plot.png")
if fi_path:
    print("  model_feature_importance.png")
print("  trained_aqi_model.joblib")
