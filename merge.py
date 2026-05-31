import pandas as pd

wide = pd.read_csv('knmi_all_stations_wide.csv', parse_dates=['date'])
ned  = pd.read_csv('ned_wind_daily.csv', parse_dates=['date'])

ned['date'] = ned['date'] + pd.Timedelta(days=1)

merged = wide.merge(ned, on='date', how='inner')

print(f'Rows: {len(merged)}')
print(f'Columns: {len(merged.columns)}')
print(f'Date range: {merged.date.min().date()} to {merged.date.max().date()}')
print(f'Missing: {merged.isnull().sum().sum()}')
print(merged.head(3).to_string())

merged.to_csv('nl_wind_energy_final.csv', index=False)
print('Saved: nl_wind_energy_final.csv')