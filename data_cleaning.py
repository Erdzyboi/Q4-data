# ════════════════════════════════════════════════════════
#                   DATA CLEANING
#         Group 2 - Wind Energy Forecasting Q4
#         Input:  nl_wind_energy_final.csv
#         Output: nl_wind_energy_clean.csv
# ════════════════════════════════════════════════════════

import pandas as pd
import numpy as np

# ── Step 1: Load dataset ─────────────────────────────────
df = pd.read_csv('nl_wind_energy_final.csv', parse_dates=['date'])
print(f"Shape: {df.shape}")
print(f"Date range: {df['date'].min().date()} to {df['date'].max().date()}")

# ── Step 2: Remove duplicate rows ────────────────────────
before = len(df)
df = df.drop_duplicates(subset=['date'], keep='first').reset_index(drop=True)
print(f"\nDuplicates removed: {before - len(df)} rows dropped")

# ── Step 3: Replace impossible values with NaN ───────────
checks = {
    'wind_speed_ms_de_bilt':             (-1, 35),
    'wind_speed_ms_de_kooy':             (-1, 35),
    'wind_speed_ms_eelde':               (-1, 35),
    'wind_speed_ms_hoek_van_holland':    (-1, 35),
    'wind_speed_ms_vlissingen':          (-1, 35),
    'temperature_c_de_bilt':             (-25, 45),
    'temperature_c_de_kooy':             (-25, 45),
    'temperature_c_eelde':               (-25, 45),
    'temperature_c_hoek_van_holland':    (-25, 45),
    'temperature_c_vlissingen':          (-25, 45),
    'wind_energy_gwh':                   (-1, 500),
    'wind_onshore_gwh':                  (-1, 300),
    'wind_offshore_gwh':                 (-1, 300),
    'sunshine_hours_de_bilt':            (-1, 17),
    'sunshine_hours_de_kooy':            (-1, 17),
    'sunshine_hours_eelde':              (-1, 17),
    'sunshine_hours_hoek_van_holland':   (-1, 17),
    'sunshine_hours_vlissingen':         (-1, 17),
}

for col, (lo, hi) in checks.items():
    if col not in df.columns:
        continue
    bad = ((df[col] < lo) | (df[col] > hi)).sum()
    df.loc[(df[col] < lo) | (df[col] > hi), col] = np.nan
    if bad > 0:
        print(f"Replaced {bad} impossible value(s) in {col}")

# ── Step 4: Fill missing values ──────────────────────────
# Interpolate continuous weather measurements
weather_cols = [c for c in df.columns if c != 'date' and
                not c.endswith('_gwh')]
df[weather_cols] = df[weather_cols].interpolate(method='linear')

# Energy columns - interpolate (real measurements, not infrastructure)
df['wind_energy_gwh']   = df['wind_energy_gwh'].interpolate(method='linear')
df['wind_onshore_gwh']  = df['wind_onshore_gwh'].interpolate(method='linear')
df['wind_offshore_gwh'] = df['wind_offshore_gwh'].interpolate(method='linear')

print(f"\nMissing values after filling: {df.isnull().sum().sum()}")

# ── Step 5: Validate final dataset ───────────────────────
print(f"\nFinal dataset: {len(df)} rows, {len(df.columns)} columns")
print(f"Date range: {df['date'].min().date()} to {df['date'].max().date()}")
print(f"Expected 1095 rows: {len(df) == 1095}")
print(f"\nSummary statistics:")
print(df[['wind_speed_ms_de_bilt', 'wind_speed_ms_hoek_van_holland',
          'wind_energy_gwh', 'wind_onshore_gwh', 'wind_offshore_gwh']].describe().round(2).to_string())

# ── Step 6: Save ─────────────────────────────────────────
df = df.sort_values('date').reset_index(drop=True)
df.to_csv('nl_wind_energy_clean.csv', index=False)
print("\nSaved: nl_wind_energy_clean.csv")