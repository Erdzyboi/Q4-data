# ════════════════════════════════════════════════════════
#                   DATA CLEANING
# ════════════════════════════════════════════════════════

import pandas as pd
import numpy as np

# Load raw dataset
df = pd.read_csv('nl_wind_energy_DIRTY.csv')
print(f"Shape: {df.shape}")
print(f"Columns: {list(df.columns)}")

# ── Step 1: Fix date type ────────────────────────────────
df['date'] = pd.to_datetime(df['date'])
print(f"\nDate type fixed: {df['date'].dtype}")

# ── Step 2: Remove extra spaces ──────────────────────────
df['station'] = df['station'].str.strip()
print(f"Station names cleaned.")

# ── Step 3: Remove duplicate rows ────────────────────────
before = len(df)
df = df.drop_duplicates(subset=['date'], keep='first').reset_index(drop=True)
print(f"\nDuplicates removed: {before - len(df)} rows dropped")

# ── Step 4: Replace impossible values with NaN ───────────
checks = {
    'wind_speed_ms':       (0, 35),
    'temperature_c':       (-25, 45),
    'wind_energy_gwh':     (0, 300),
    'capacity_factor_pct': (0, 100),
    'sunshine_hours':      (0, 17),
}
for col, (lo, hi) in checks.items():
    bad = ((df[col] < lo) | (df[col] > hi)).sum()
    df.loc[(df[col] < lo) | (df[col] > hi), col] = np.nan
    if bad > 0:
        print(f"Replaced {bad} impossible value(s) in {col}")

# ── Step 5: Fill missing values ──────────────────────────
numeric_cols = ['wind_speed_ms','wind_direction_deg','temperature_c',
                'air_pressure_hpa','sunshine_hours','precipitation_mm',
                'wind_energy_gwh','capacity_factor_pct','installed_capacity_gw']
df[numeric_cols] = df[numeric_cols].interpolate(method='linear')
print(f"\nMissing values after filling: {df.isnull().sum().sum()}")

# ── Final result ──────────────────────────────────────────
df = df.sort_values('date').reset_index(drop=True)
print(f"\nFinal clean dataset: {len(df)} rows, {df.isnull().sum().sum()} nulls")
df.to_csv('nl_wind_energy_CLEAN.csv', index=False)
print("Saved: nl_wind_energy_CLEAN.csv")
