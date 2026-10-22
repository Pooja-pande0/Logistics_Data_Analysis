"""
Week 2: Logistics data cleaning and preprocessing
Input: dynamic_supply_chain_logistics_dataset.csv
Outputs:
  - cleaned_logistics_data.csv
  - outlier_summary.csv

Install dependencies:
  pip install pandas scikit-learn
Run:
  python preprocessing.py
"""
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

INPUT_FILE = "dynamic_supply_chain_logistics_dataset.csv"
OUTPUT_FILE = "cleaned_logistics_data.csv"
OUTLIER_FILE = "outlier_summary.csv"

df = pd.read_csv(INPUT_FILE)
print("Original dataset shape:", df.shape)
print("Missing values by column before cleaning:")
print(df.isna().sum())
print("Exact duplicate rows:", df.duplicated().sum())

# Parse the timestamp column into a datetime type.
df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")

# Remove exact duplicate rows only; repeated timestamps alone are not assumed duplicates.
df = df.drop_duplicates().copy()

# Convert expected numeric columns to numeric values.
numeric_cols = [
    "vehicle_gps_latitude", "vehicle_gps_longitude",
    "fuel_consumption_rate", "eta_variation_hours",
    "traffic_congestion_level", "warehouse_inventory_level",
    "loading_unloading_time", "handling_equipment_availability",
    "order_fulfillment_status", "weather_condition_severity",
    "port_congestion_level", "shipping_costs",
    "supplier_reliability_score", "lead_time_days",
    "historical_demand", "iot_temperature", "cargo_condition_status",
    "route_risk_level", "customs_clearance_time",
    "driver_behavior_score", "fatigue_monitoring_score",
    "disruption_likelihood_score", "delay_probability",
    "delivery_time_deviation"
]
for col in numeric_cols:
    df[col] = pd.to_numeric(df[col], errors="coerce")

# Handle missing values only if any are present after type conversion.
# Median is used for numeric fields; categorical values use the mode when available.
for col in numeric_cols:
    if df[col].isna().any() and df[col].notna().any():
        df[col] = df[col].fillna(df[col].median())

if df["risk_classification"].isna().any() and not df["risk_classification"].mode().empty:
    df["risk_classification"] = df["risk_classification"].fillna(
        df["risk_classification"].mode().iloc[0]
    )

# Identify potential outliers with the IQR rule. Flag; do not delete automatically.
outlier_columns = ["shipping_costs", "lead_time_days", "delivery_time_deviation"]
summary = []
for col in outlier_columns:
    q1 = df[col].quantile(0.25)
    q3 = df[col].quantile(0.75)
    iqr = q3 - q1
    lower = q1 - 1.5 * iqr
    upper = q3 + 1.5 * iqr
    mask = (df[col] < lower) | (df[col] > upper)
    df[col + "_outlier_flag"] = mask
    summary.append({
        "column": col, "Q1": q1, "Q3": q3, "IQR": iqr,
        "lower_bound": lower, "upper_bound": upper,
        "potential_outlier_count": int(mask.sum())
    })
pd.DataFrame(summary).to_csv(OUTLIER_FILE, index=False)

# Normalize selected continuous features to [0, 1].
# GPS coordinates, timestamp, and categorical risk label are not scaled.
features = [
    "fuel_consumption_rate", "eta_variation_hours",
    "traffic_congestion_level", "warehouse_inventory_level",
    "loading_unloading_time", "shipping_costs", "lead_time_days",
    "historical_demand", "delivery_time_deviation"
]
# Only scale rows complete across the selected features.
usable = df[features].notna().all(axis=1)
scaler = MinMaxScaler()
scaled_names = [f"{col}_normalized" for col in features]
df.loc[usable, scaled_names] = scaler.fit_transform(df.loc[usable, features])

df.to_csv(OUTPUT_FILE, index=False)
print("Cleaned dataset shape:", df.shape)
print("Missing values after cleaning:")
print(df.isna().sum())
print("Saved:", OUTPUT_FILE)
print("Saved:", OUTLIER_FILE)
