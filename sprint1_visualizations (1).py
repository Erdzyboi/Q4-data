# ════════════════════════════════════════════════════════
#           SPRINT 1 — VISUALIZATIONS
#     Wind Energy Forecasting — Netherlands 2021-2023
# ════════════════════════════════════════════════════════

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

# ── Load clean dataset ───────────────────────────────────
df = pd.read_csv('nl_wind_energy_clean.csv')
df['date'] = pd.to_datetime(df['date'])

# ── Feature engineering ──────────────────────────────────
# Use Hoek van Holland as primary wind speed (strongest coastal station)
df['wind_speed_ms']  = df['wind_speed_ms_330']   # Hoek van Holland
df['temperature_c']  = df['temperature_c_260']   # De Bilt
df['air_pressure_hpa'] = df['air_pressure_hpa_260']
df['wind_cubed']     = df['wind_speed_ms'] ** 3
df['is_sw_wind']     = ((df['wind_direction_deg_330'] >= 180) &
                        (df['wind_direction_deg_330'] <= 270)).astype(int)

# Season labels
def get_season(month):
    if month in [12, 1, 2]:  return 'Winter'
    elif month in [3, 4, 5]: return 'Spring'
    elif month in [6, 7, 8]: return 'Summer'
    else:                     return 'Autumn'

df['season'] = df['date'].dt.month.apply(get_season)
df['month']  = df['date'].dt.month

# ── Train Linear Regression model ───────────────────────
features = ['wind_speed_ms', 'wind_cubed', 'temperature_c',
            'air_pressure_hpa', 'is_sw_wind']
X = df[features]
y = df['wind_energy_gwh']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

model = LinearRegression()
model.fit(X_train, y_train)
df['predicted_gwh'] = model.predict(X)

# ── Station average wind speeds ──────────────────────────
stations = {
    'De Bilt\n(260)':           df['wind_speed_ms_260'].mean(),
    'De Kooy\n(235)':           df['wind_speed_ms_235'].mean(),
    'Hoek v. Holland\n(330)':   df['wind_speed_ms_330'].mean(),
    'Vlissingen\n(310)':        df['wind_speed_ms_310'].mean(),
    'Eelde\n(280)':             df['wind_speed_ms_280'].mean(),
}
station_names  = list(stations.keys())
station_speeds = list(stations.values())

# ════════════════════════════════════════════════════════
#   FIGURE — 4 charts in 2x2 grid
# ════════════════════════════════════════════════════════
fig = plt.figure(figsize=(16, 12))
fig.suptitle('Sprint 1 — Wind Energy Forecasting Netherlands (2021–2023)',
             fontsize=14, fontweight='bold', y=0.98)

gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.4, wspace=0.35)

# ── CHART (a): Wind Speed vs Energy Output ───────────────
ax1 = fig.add_subplot(gs[0, 0])

season_colors = {'Winter': '#2C7BB6', 'Spring': '#74C476',
                 'Summer': '#FD8D3C', 'Autumn': '#D7301F'}
for season, color in season_colors.items():
    mask = df['season'] == season
    ax1.scatter(df.loc[mask, 'wind_speed_ms'],
                df.loc[mask, 'wind_energy_gwh'],
                c=color, alpha=0.5, s=18, label=season)

ax1.set_xlabel('Daily Avg Wind Speed — Hoek v. Holland (m/s)', fontsize=9)
ax1.set_ylabel('Wind Energy Output (GWh/day)', fontsize=9)
ax1.set_title('(a) Wind Speed vs Energy Output', fontweight='bold', fontsize=10)
ax1.legend(fontsize=8, markerscale=1.2)

corr = df['wind_speed_ms'].corr(df['wind_energy_gwh'])
ax1.text(0.05, 0.92, f'r = {corr:.3f}',
         transform=ax1.transAxes, fontsize=9,
         bbox=dict(boxstyle='round,pad=0.3', facecolor='lightyellow'))

# ── CHART (b): Average Wind Speed per Station ────────────
ax2 = fig.add_subplot(gs[0, 1])

bar_colors = ['#4575B4', '#74ADD1', '#FD8D3C', '#D73027', '#1A9850']
bars = ax2.bar(station_names, station_speeds, color=bar_colors,
               edgecolor='white', linewidth=0.8)

for bar, val in zip(bars, station_speeds):
    ax2.text(bar.get_x() + bar.get_width() / 2,
             bar.get_height() + 0.05,
             f'{val:.2f}', ha='center', va='bottom', fontsize=8)

ax2.set_ylabel('Average Wind Speed (m/s)', fontsize=9)
ax2.set_title('(b) Avg Wind Speed per KNMI Station', fontweight='bold', fontsize=10)
ax2.set_ylim(0, max(station_speeds) * 1.2)
ax2.axhline(np.mean(station_speeds), color='red', linestyle='--',
            linewidth=1, label=f'Avg {np.mean(station_speeds):.2f} m/s')
ax2.legend(fontsize=8)

# ── CHART (c): Actual vs Predicted Energy Over Time ──────
ax3 = fig.add_subplot(gs[1, 0])

ax3.fill_between(df['date'], df['wind_energy_gwh'],
                 alpha=0.4, color='#2C7BB6', label='Actual')
ax3.plot(df['date'], df['predicted_gwh'],
         color='#D73027', linewidth=0.8, linestyle='--', label='Predicted')

ax3.set_xlabel('Date', fontsize=9)
ax3.set_ylabel('Wind Energy (GWh/day)', fontsize=9)
ax3.set_title('(c) Actual vs Predicted Wind Energy (2021–2023)',
              fontweight='bold', fontsize=10)
ax3.legend(fontsize=8)

r2 = model.score(X_test, y_test)
ax3.text(0.02, 0.92, f'R² = {r2:.3f}',
         transform=ax3.transAxes, fontsize=9,
         bbox=dict(boxstyle='round,pad=0.3', facecolor='lightyellow'))

# ── CHART (d): Energy Output by Season (box plot) ────────
ax4 = fig.add_subplot(gs[1, 1])

season_order  = ['Winter', 'Spring', 'Summer', 'Autumn']
season_data   = [df[df['season'] == s]['wind_energy_gwh'].values
                 for s in season_order]
season_cols   = ['#2C7BB6', '#74C476', '#FD8D3C', '#D7301F']

bp = ax4.boxplot(season_data, labels=season_order, patch_artist=True,
                 medianprops=dict(color='black', linewidth=2))

for patch, color in zip(bp['boxes'], season_cols):
    patch.set_facecolor(color)
    patch.set_alpha(0.7)

ax4.set_ylabel('Wind Energy Output (GWh/day)', fontsize=9)
ax4.set_title('(d) Energy Output Distribution by Season',
              fontweight='bold', fontsize=10)

for i, (season, data) in enumerate(zip(season_order, season_data)):
    ax4.text(i + 1, np.median(data) + 1,
             f'med={np.median(data):.1f}',
             ha='center', fontsize=7.5, color='black')

# ── Save and show ─────────────────────────────────────────
plt.savefig('sprint1_visualizations.png', dpi=150, bbox_inches='tight')
plt.show()
print("Saved: sprint1_visualizations.png")
