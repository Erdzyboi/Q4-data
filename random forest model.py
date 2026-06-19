import os
import sys
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error

CSV_FALLBACK_PATH = "nl_wind_energy_clean.csv"
MONGO_DB_NAME = "wind_energy_q4"
MONGO_COLLECTION = "wind_energy_daily"

RAW_STATION_WIND_COLS = [
    "wind_speed_ms_de_bilt",
    "wind_speed_ms_de_kooy",
    "wind_speed_ms_eelde",
    "wind_speed_ms_hoek_van_holland",
    "wind_speed_ms_vlissingen",
]


def load_from_mongodb():
    try:
        from pymongo import MongoClient
        from dotenv import load_dotenv
        load_dotenv()

        uri = os.getenv("MONGODB_URI")
        if not uri:
            print("[MongoDB] No MONGODB_URI found in .env — using CSV fallback.")
            return None

        client = MongoClient(uri, serverSelectionTimeoutMS=5000)
        client.admin.command("ping")

        db = client[MONGO_DB_NAME]
        collection = db[MONGO_COLLECTION]

        records = list(collection.find({}, {"_id": 0}))
        if not records:
            print(f"[MongoDB] Connected, but '{MONGO_COLLECTION}' is empty — using CSV fallback.")
            return None

        df = pd.DataFrame(records)
        print(f"[MongoDB] Loaded {len(df)} records from {MONGO_DB_NAME}.{MONGO_COLLECTION}")
        return df

    except Exception as e:
        print(f"[MongoDB] Could not load data ({e}) — using CSV fallback.")
        return None


def load_data():
    df = load_from_mongodb()
    if df is None:
        if not os.path.exists(CSV_FALLBACK_PATH):
            sys.exit(f"ERROR: No MongoDB data and no fallback CSV '{CSV_FALLBACK_PATH}' found.")
        df = pd.read_csv(CSV_FALLBACK_PATH)
        print(f"[CSV] Loaded {len(df)} records from {CSV_FALLBACK_PATH}")
    return df


df = load_data()
df["date"] = pd.to_datetime(df["date"])

if "wind_speed_avg" not in df.columns:
    missing_station_cols = [c for c in RAW_STATION_WIND_COLS if c not in df.columns]
    if missing_station_cols:
        sys.exit(f"ERROR: Missing columns needed to compute wind_speed_avg: {missing_station_cols}")
    df["wind_speed_avg"] = df[RAW_STATION_WIND_COLS].mean(axis=1)

if "wind_cubed" not in df.columns:
    df["wind_cubed"] = df["wind_speed_avg"] ** 3

if "is_sw_wind" not in df.columns:
    if "wind_direction_deg_hoek_van_holland" not in df.columns:
        sys.exit("ERROR: Cannot compute is_sw_wind — wind_direction_deg_hoek_van_holland missing.")
    df["is_sw_wind"] = (
        (df["wind_direction_deg_hoek_van_holland"] >= 180)
        & (df["wind_direction_deg_hoek_van_holland"] <= 270)
    ).astype(int)

features = [
    "wind_speed_avg",
    "wind_cubed",
    "temperature_c_de_bilt",
    "air_pressure_hpa_de_bilt",
    "is_sw_wind",
]
feature_labels = [
    "Avg Wind Speed",
    "Wind Speed Cubed",
    "Temperature (De Bilt)",
    "Air Pressure (De Bilt)",
    "SW Wind Direction",
]

missing_features = [f for f in features if f not in df.columns]
if missing_features:
    sys.exit(f"ERROR: Required feature columns missing: {missing_features}")

X = df[features]
y = df["wind_energy_gwh"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

lr = LinearRegression()
lr.fit(X_train, y_train)
lr_pred = lr.predict(X_test)
lr_r2 = r2_score(y_test, lr_pred)
lr_mae = mean_absolute_error(y_test, lr_pred)

rf = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
rf.fit(X_train, y_train)
rf_pred = rf.predict(X_test)
rf_r2 = r2_score(y_test, rf_pred)
rf_mae = mean_absolute_error(y_test, rf_pred)

gb = GradientBoostingRegressor(random_state=42)
gb.fit(X_train, y_train)
gb_pred = gb.predict(X_test)
gb_r2 = r2_score(y_test, gb_pred)
gb_mae = mean_absolute_error(y_test, gb_pred)

print()
print("=" * 55)
print("MODEL EVALUATION RESULTS")
print("=" * 55)
print(f"Linear Regression     |  R²: {lr_r2:.4f}  |  MAE: {lr_mae:.4f} GWh")
print(f"Random Forest (final) |  R²: {rf_r2:.4f}  |  MAE: {rf_mae:.4f} GWh")
print(f"Gradient Boosting     |  R²: {gb_r2:.4f}  |  MAE: {gb_mae:.4f} GWh  (alternative, not used)")
print()
print("Feature Importances (Random Forest):")
for name, imp in sorted(zip(feature_labels, rf.feature_importances_), key=lambda x: -x[1]):
    print(f"  {name:<28} {imp:.4f}")

joblib.dump(rf, "random_forest_model.joblib")
joblib.dump(features, "random_forest_features.joblib")
print("\nSaved: random_forest_model.joblib")

fig, axes = plt.subplots(1, 2, figsize=(14, 6))
fig.suptitle(
    "Sprint 3 (Final) — Random Forest Model Results\n"
    "Wind Energy Forecasting Netherlands (2021–2023)",
    fontsize=13, fontweight="bold",
)

importances = rf.feature_importances_
sorted_idx = np.argsort(importances)
colors = ["#2C7BB6", "#74ADD1", "#FD8D3C", "#D73027", "#1A9850"]

bars = axes[0].barh(
    [feature_labels[i] for i in sorted_idx],
    importances[sorted_idx],
    color=[colors[i] for i in sorted_idx],
    edgecolor="white",
)
for bar, val in zip(bars, importances[sorted_idx]):
    axes[0].text(val + 0.005, bar.get_y() + bar.get_height() / 2, f"{val:.3f}", va="center", fontsize=9)
axes[0].set_xlabel("Feature Importance (Gini)", fontsize=10)
axes[0].set_title("(a) Random Forest Feature Importance", fontweight="bold", fontsize=11)
axes[0].set_xlim(0, max(importances) * 1.2)

models = ["Linear Regression\n(Sprint 1)", "Random Forest\n(final)", "Gradient Boosting\n(alternative)"]
r2_scores = [lr_r2, rf_r2, gb_r2]
mae_scores = [lr_mae, rf_mae, gb_mae]

x = np.arange(len(models))
w = 0.35
ax2 = axes[1]
ax2b = ax2.twinx()

b1 = ax2.bar(x - w / 2, r2_scores, w, label="R²", color="#2C7BB6", alpha=0.85)
b2 = ax2b.bar(x + w / 2, mae_scores, w, label="MAE (GWh)", color="#FD8D3C", alpha=0.85)

ax2.set_ylabel("R² Score", fontsize=10, color="#2C7BB6")
ax2b.set_ylabel("MAE (GWh/day)", fontsize=10, color="#FD8D3C")
ax2.set_xticks(x)
ax2.set_xticklabels(models, fontsize=9)
ax2.set_ylim(0.80, 0.92)
ax2b.set_ylim(10, 17)
ax2.set_title("(b) Model Comparison: R² and MAE", fontweight="bold", fontsize=11)

for bar, val in zip(b1, r2_scores):
    ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.001, f"{val:.4f}", ha="center", fontsize=8, color="#2C7BB6")
for bar, val in zip(b2, mae_scores):
    ax2b.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.1, f"{val:.2f}", ha="center", fontsize=8, color="#D73027")

lines1, labels1 = ax2.get_legend_handles_labels()
lines2, labels2 = ax2b.get_legend_handles_labels()
ax2.legend(lines1 + lines2, labels1 + labels2, fontsize=9, loc="lower right")

plt.tight_layout()
plt.savefig("sprint3_model_results.png", dpi=150, bbox_inches="tight")
plt.show()
print("Saved: sprint3_model_results.png")

results_df = pd.DataFrame({
    "date": df.loc[X_test.index, "date"].values,
    "actual_gwh": y_test.values,
    "predicted_gwh": rf_pred,
}).sort_values("date").reset_index(drop=True)
results_df["abs_error_gwh"] = (results_df["actual_gwh"] - results_df["predicted_gwh"]).abs()

print()
print("=" * 60)
print("SAMPLE PREDICTIONS vs ACTUAL (first 10 test days)")
print("=" * 60)
print(results_df.head(10).to_string(index=False))
print()
print(f"Mean actual output:    {results_df['actual_gwh'].mean():.2f} GWh/day")
print(f"Mean predicted output: {results_df['predicted_gwh'].mean():.2f} GWh/day")
print(f"Mean absolute error:   {results_df['abs_error_gwh'].mean():.2f} GWh/day")

results_df.to_csv("rf_predictions_vs_actual.csv", index=False)
print("\nSaved: rf_predictions_vs_actual.csv")
