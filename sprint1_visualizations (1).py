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

# ── Load dataset ─────────────────────────────────────────
df = pd.read_csv("nl_wind_energy_clean.csv")
df["date"] = pd.to_datetime(df["date"])

# ── Feature Engineering ──────────────────────────────────
df["wind_speed_ms"] = df["wind_speed_ms_hoek_van_holland"]
df["temperature_c"] = df["temperature_c_de_bilt"]
df["air_pressure_hpa"] = df["air_pressure_hpa_de_bilt"]

df["wind_cubed"] = df["wind_speed_ms"] ** 3

df["is_sw_wind"] = (
    (df["wind_direction_deg_hoek_van_holland"] >= 180)
    & (df["wind_direction_deg_hoek_van_holland"] <= 270)
).astype(int)

# ── Seasons ──────────────────────────────────────────────
def get_season(month):
    if month in [12, 1, 2]:
        return "Winter"
    elif month in [3, 4, 5]:
        return "Spring"
    elif month in [6, 7, 8]:
        return "Summer"
    else:
        return "Autumn"

df["season"] = df["date"].dt.month.apply(get_season)

# ── Linear Regression Model ──────────────────────────────
features = [
    "wind_speed_ms",
    "wind_cubed",
    "temperature_c",
    "air_pressure_hpa",
    "is_sw_wind"
]

X = df[features]
y = df["wind_energy_gwh"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

model = LinearRegression()
model.fit(X_train, y_train)

df["predicted_gwh"] = model.predict(X)

# ── Station Comparison ───────────────────────────────────
stations = {
    "De Bilt": df["wind_speed_ms_de_bilt"].mean(),
    "De Kooy": df["wind_speed_ms_de_kooy"].mean(),
    "Hoek van Holland": df["wind_speed_ms_hoek_van_holland"].mean(),
    "Vlissingen": df["wind_speed_ms_vlissingen"].mean(),
    "Eelde": df["wind_speed_ms_eelde"].mean()
}

station_names = list(stations.keys())
station_speeds = list(stations.values())

# ── Figure Layout ────────────────────────────────────────
fig = plt.figure(figsize=(16, 12))

fig.suptitle(
    "Sprint 1 — Wind Energy Forecasting Netherlands (2021–2023)",
    fontsize=16,
    fontweight="bold"
)

gs = gridspec.GridSpec(
    2,
    2,
    figure=fig,
    hspace=0.35,
    wspace=0.3
)

# ─────────────────────────────────────────────────────────
# Chart A — Wind Speed vs Energy Output
# ─────────────────────────────────────────────────────────
ax1 = fig.add_subplot(gs[0, 0])

season_colors = {
    "Winter": "#2C7BB6",
    "Spring": "#74C476",
    "Summer": "#FD8D3C",
    "Autumn": "#D7301F"
}

for season, color in season_colors.items():
    mask = df["season"] == season

    ax1.scatter(
        df.loc[mask, "wind_speed_ms"],
        df.loc[mask, "wind_energy_gwh"],
        color=color,
        alpha=0.5,
        s=20,
        label=season
    )

corr = df["wind_speed_ms"].corr(df["wind_energy_gwh"])

ax1.set_title(
    "(a) Wind Speed vs Energy Output",
    fontweight="bold"
)

ax1.set_xlabel(
    "Wind Speed (m/s)"
)

ax1.set_ylabel(
    "Energy Output (GWh)"
)

ax1.legend()

ax1.text(
    0.05,
    0.92,
    f"r = {corr:.3f}",
    transform=ax1.transAxes,
    bbox=dict(facecolor="lightyellow")
)

# ─────────────────────────────────────────────────────────
# Chart B — Station Wind Speeds
# ─────────────────────────────────────────────────────────
ax2 = fig.add_subplot(gs[0, 1])

bars = ax2.bar(
    station_names,
    station_speeds
)

for bar, value in zip(bars, station_speeds):
    ax2.text(
        bar.get_x() + bar.get_width() / 2,
        value + 0.05,
        f"{value:.2f}",
        ha="center"
    )

ax2.set_title(
    "(b) Average Wind Speed per Station",
    fontweight="bold"
)

ax2.set_ylabel(
    "Wind Speed (m/s)"
)

# ─────────────────────────────────────────────────────────
# Chart C — Actual vs Predicted
# ─────────────────────────────────────────────────────────
ax3 = fig.add_subplot(gs[1, 0])

ax3.plot(
    df["date"],
    df["wind_energy_gwh"],
    label="Actual",
    linewidth=1
)

ax3.plot(
    df["date"],
    df["predicted_gwh"],
    linestyle="--",
    linewidth=1,
    label="Predicted"
)

r2 = model.score(X_test, y_test)

ax3.text(
    0.05,
    0.92,
    f"R² = {r2:.3f}",
    transform=ax3.transAxes,
    bbox=dict(facecolor="lightyellow")
)

ax3.set_title(
    "(c) Actual vs Predicted Energy Output",
    fontweight="bold"
)

ax3.set_ylabel(
    "Energy Output (GWh)"
)

ax3.legend()

# ─────────────────────────────────────────────────────────
# Chart D — Seasonal Distribution
# ─────────────────────────────────────────────────────────
ax4 = fig.add_subplot(gs[1, 1])

season_order = [
    "Winter",
    "Spring",
    "Summer",
    "Autumn"
]

season_data = [
    df[df["season"] == season]["wind_energy_gwh"]
    for season in season_order
]

box = ax4.boxplot(
    season_data,
    tick_labels=season_order,
    patch_artist=True
)

colors = [
    "#2C7BB6",
    "#74C476",
    "#FD8D3C",
    "#D7301F"
]

for patch, color in zip(box["boxes"], colors):
    patch.set_facecolor(color)
    patch.set_alpha(0.7)

ax4.set_title(
    "(d) Seasonal Energy Output Distribution",
    fontweight="bold"
)

ax4.set_ylabel(
    "Energy Output (GWh)"
)

# ── Save ─────────────────────────────────────────────────
plt.savefig(
    "sprint1_visualizations.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()

print("SUCCESS")
print("Saved: sprint1_visualizations.png")