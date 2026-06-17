# ════════════════════════════════════════════════════════
#           SPRINT 3 — RANDOM FOREST MODEL
#     Wind Energy Forecasting — Netherlands 2021-2023
#     Student 3: Raigardas Mozgeris
# ════════════════════════════════════════════════════════

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error

# ── Load dataset ─────────────────────────────────────────
df = pd.read_csv('nl_wind_energy_clean.csv')
df['date'] = pd.to_datetime(df['date'])

# ── Feature engineering ──────────────────────────────────
df['wind_speed_avg'] = df[[
    'wind_speed_ms_de_bilt',
    'wind_speed_ms_de_kooy',
    'wind_speed_ms_eelde',
    'wind_speed_ms_hoek_van_holland',
    'wind_speed_ms_vlissingen'
]].mean(axis=1)

df['wind_cubed'] = df['wind_speed_avg'] ** 3

df['is_sw_wind'] = (
    (df['wind_direction_deg_hoek_van_holland'] >= 180) &
    (df['wind_direction_deg_hoek_van_holland'] <= 270)
).astype(int)

# ── Train / test split ───────────────────────────────────
features = [
    'wind_speed_avg',
    'wind_cubed',
    'temperature_c_de_bilt',
    'air_pressure_hpa_de_bilt',
    'is_sw_wind'
]
feature_labels = [
    'Avg Wind Speed',
    'Wind Speed Cubed',
    'Temperature (De Bilt)',
    'Air Pressure (De Bilt)',
    'SW Wind Direction'
]

X = df[features]
y = df['wind_energy_gwh']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ── Train Linear Regression (Sprint 1 baseline) ──────────
lr = LinearRegression()
lr.fit(X_train, y_train)
lr_pred = lr.predict(X_test)
lr_r2  = r2_score(y_test, lr_pred)
lr_mae = mean_absolute_error(y_test, lr_pred)

# ── Train Random Forest ──────────────────────────────────
rf = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
rf.fit(X_train, y_train)
rf_pred = rf.predict(X_test)
rf_r2  = r2_score(y_test, rf_pred)
rf_mae = mean_absolute_error(y_test, rf_pred)

# ── Print results ─────────────────────────────────────────
print("=" * 45)
print("MODEL EVALUATION RESULTS")
print("=" * 45)
print(f"Linear Regression  |  R²: {lr_r2:.4f}  |  MAE: {lr_mae:.4f} GWh")
print(f"Random Forest      |  R²: {rf_r2:.4f}  |  MAE: {rf_mae:.4f} GWh")
print()
print("Feature Importances (Random Forest):")
for name, imp in sorted(zip(feature_labels, rf.feature_importances_), key=lambda x: -x[1]):
    print(f"  {name:<28} {imp:.4f}")

# ── Plot ──────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 6))
fig.suptitle(
    'Sprint 3 — Random Forest Model Results\nWind Energy Forecasting Netherlands (2021–2023)',
    fontsize=13, fontweight='bold'
)

# Chart (a): Feature Importance
importances = rf.feature_importances_
sorted_idx  = np.argsort(importances)
colors = ['#2C7BB6', '#74ADD1', '#FD8D3C', '#D73027', '#1A9850']

bars = axes[0].barh(
    [feature_labels[i] for i in sorted_idx],
    importances[sorted_idx],
    color=[colors[i] for i in sorted_idx],
    edgecolor='white'
)
for bar, val in zip(bars, importances[sorted_idx]):
    axes[0].text(
        val + 0.005, bar.get_y() + bar.get_height() / 2,
        f'{val:.3f}', va='center', fontsize=9
    )
axes[0].set_xlabel('Feature Importance (Gini)', fontsize=10)
axes[0].set_title('(a) Random Forest Feature Importance', fontweight='bold', fontsize=11)
axes[0].set_xlim(0, max(importances) * 1.2)

# Chart (b): Model Comparison
models    = ['Linear Regression\n(Sprint 1)', 'Random Forest\n(Sprint 2)']
r2_scores = [lr_r2, rf_r2]
mae_scores = [lr_mae, rf_mae]

x  = np.arange(len(models))
w  = 0.35
ax2  = axes[1]
ax2b = ax2.twinx()

b1 = ax2.bar(x - w / 2,  r2_scores,  w, label='R²',          color='#2C7BB6', alpha=0.85)
b2 = ax2b.bar(x + w / 2, mae_scores, w, label='MAE (GWh)',   color='#FD8D3C', alpha=0.85)

ax2.set_ylabel('R² Score',        fontsize=10, color='#2C7BB6')
ax2b.set_ylabel('MAE (GWh/day)', fontsize=10, color='#FD8D3C')
ax2.set_xticks(x)
ax2.set_xticklabels(models, fontsize=10)
ax2.set_ylim(0.80, 0.92)
ax2b.set_ylim(10, 17)
ax2.set_title('(b) Model Comparison: R² and MAE', fontweight='bold', fontsize=11)

for bar, val in zip(b1, r2_scores):
    ax2.text(bar.get_x() + bar.get_width() / 2,
             bar.get_height() + 0.001,
             f'{val:.4f}', ha='center', fontsize=9, color='#2C7BB6')
for bar, val in zip(b2, mae_scores):
    ax2b.text(bar.get_x() + bar.get_width() / 2,
              bar.get_height() + 0.1,
              f'{val:.2f}', ha='center', fontsize=9, color='#D73027')

lines1, labels1 = ax2.get_legend_handles_labels()
lines2, labels2 = ax2b.get_legend_handles_labels()
ax2.legend(lines1 + lines2, labels1 + labels2, fontsize=9, loc='lower right')

plt.tight_layout()
plt.savefig('sprint3_model_results.png', dpi=150, bbox_inches='tight')
plt.show()
print("\nSaved: sprint3_model_results.png")
