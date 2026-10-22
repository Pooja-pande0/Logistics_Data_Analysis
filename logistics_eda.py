"""
Logistics EDA and visualization
Input: cleaned_logistics_data.csv (place this file beside the script)
Outputs: logistics_charts/*.png and eda_summary.txt
Install: pip install pandas matplotlib
Run: python logistics_eda.py
Note: confirm column definitions, units, and dataset provenance before business use.
"""
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

INPUT = Path("cleaned_logistics_data.csv")
OUT = Path("logistics_charts")
OUT.mkdir(exist_ok=True)
df = pd.read_csv(INPUT)
if "timestamp" in df:
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")

numeric = [
    "fuel_consumption_rate", "eta_variation_hours", "traffic_congestion_level",
    "warehouse_inventory_level", "loading_unloading_time", "shipping_costs",
    "supplier_reliability_score", "lead_time_days", "historical_demand",
    "customs_clearance_time", "disruption_likelihood_score", "delay_probability",
    "delivery_time_deviation"
]
numeric = [c for c in numeric if c in df.columns]
for c in numeric:
    df[c] = pd.to_numeric(df[c], errors="coerce")

print("Dataset shape:", df.shape)
print("\\nData types:\\n", df.dtypes)
print("\\nMissing values:\\n", df.isna().sum().sort_values(ascending=False))
print("\\nExact duplicate rows:", df.duplicated().sum())
print("\\nDescriptive statistics:\\n", df[numeric].describe().T)
df[numeric].describe().T.to_csv("eda_descriptive_statistics.csv")
df[numeric].corr().to_csv("eda_correlation_matrix.csv")

def save_hist(column, title, xlabel, filename):
    if column in df:
        fig, ax = plt.subplots()
        df[column].dropna().plot(kind="hist", bins=30, ax=ax)
        ax.set(title=title, xlabel=xlabel, ylabel="Number of records")
        fig.tight_layout()
        fig.savefig(OUT / filename, dpi=160)
        plt.close(fig)

save_hist("lead_time_days", "Distribution of Lead Time",
          "Lead time (days; verify units)", "lead_time_distribution.png")
save_hist("shipping_costs", "Distribution of Shipping Costs",
          "Shipping cost (units; verify source)", "shipping_costs_distribution.png")

corr_cols = [c for c in [
    "shipping_costs", "lead_time_days", "eta_variation_hours",
    "traffic_congestion_level", "fuel_consumption_rate",
    "warehouse_inventory_level", "historical_demand",
    "delay_probability", "delivery_time_deviation",
    "disruption_likelihood_score"] if c in df.columns]
corr = df[corr_cols].corr()
fig, ax = plt.subplots(figsize=(9, 7))
im = ax.imshow(corr.values, aspect="auto", vmin=-1, vmax=1)
ax.set_xticks(range(len(corr.columns)))
ax.set_xticklabels(corr.columns, rotation=65, ha="right", fontsize=8)
ax.set_yticks(range(len(corr.index)))
ax.set_yticklabels(corr.index, fontsize=8)
fig.colorbar(im, ax=ax, label="Pearson correlation")
ax.set_title("Correlation Between Selected Logistics Metrics")
fig.tight_layout()
fig.savefig(OUT / "correlation_heatmap.png", dpi=160)
plt.close(fig)

if "risk_classification" in df:
    counts = df["risk_classification"].fillna("Missing").value_counts()
    fig, ax = plt.subplots()
    counts.plot(kind="bar", ax=ax)
    ax.set(title="Records by Risk Classification",
           xlabel="Risk classification", ylabel="Number of records")
    fig.tight_layout()
    fig.savefig(OUT / "risk_classification.png", dpi=160)
    plt.close(fig)

if "timestamp" in df and "lead_time_days" in df:
    tmp = df.dropna(subset=["timestamp"]).copy()
    tmp["month"] = tmp["timestamp"].dt.to_period("M").dt.to_timestamp()
    monthly = tmp.groupby("month")["lead_time_days"].mean()
    fig, ax = plt.subplots(figsize=(9, 4.5))
    monthly.plot(ax=ax)
    ax.set(title="Average Lead Time by Month", xlabel="Month",
           ylabel="Average lead time (days; verify units)")
    fig.tight_layout()
    fig.savefig(OUT / "monthly_lead_time.png", dpi=160)
    plt.close(fig)

with open("eda_summary.txt", "w", encoding="utf-8") as f:
    f.write(f"Shape: {df.shape}\\n\\n")
    f.write("Descriptive statistics:\\n")
    f.write(df[numeric].describe().to_string())
    f.write("\\n\\nCorrelation matrix:\\n")
    f.write(df[corr_cols].corr().to_string())
print("\\nDone. Check logistics_charts/ and the CSV/TXT summary files.")
