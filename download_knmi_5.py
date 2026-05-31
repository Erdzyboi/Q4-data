import pandas as pd
import urllib.request
import zipfile
import os
from io import StringIO

STATIONS = {
    260: "De Bilt",
    235: "De Kooy",
    330: "Hoek van Holland",
    310: "Vlissingen",
    280: "Eelde"
}

def download_and_parse(station_id, station_name):
    url = f"https://cdn.knmi.nl/knmi/map/page/klimatologie/gegevens/daggegevens/etmgeg_{station_id}.zip"
    zip_path = f"knmi_{station_id}.zip"
    txt_path = f"etmgeg_{station_id}.txt"

    print(f"Downloading {station_name} ({station_id})...")
    urllib.request.urlretrieve(url, zip_path)
    with zipfile.ZipFile(zip_path, "r") as z:
        z.extractall(".")

    rows = []
    with open(txt_path, "r") as f:
        for line in f:
            line = line.strip()
            if (line.startswith(f"{station_id},2021") or
                line.startswith(f"{station_id},2022") or
                line.startswith(f"{station_id},2023")):
                rows.append(line)

    df = pd.read_csv(StringIO('\n'.join(rows)), header=None)
    df.columns = [
        'STN','YYYYMMDD','DDVEC','FHVEC','FG','FHX','FHXH','FHN','FHNH',
        'FXX','FXXH','TG','TN','TNH','TX','TXH','T10N','T10NH','SQ','SP',
        'Q','DR','RH','RHX','RHXH','PG','PX','PXH','PN','PNH','VVN','VVNH',
        'VVX','VVXH','NG','UG','UX','UXH','UN','UNH','EV24'
    ]

    def to_float(series, scale=1, clip=False):
        s = pd.to_numeric(series.astype(str).str.strip(), errors='coerce')
        if clip:
            s = s.clip(lower=0)
        return s / scale

    out = pd.DataFrame()
    out['date']               = pd.to_datetime(df['YYYYMMDD'].astype(str), format='%Y%m%d')
    out['station']            = station_name
    out['station_id']         = station_id
    out['wind_speed_ms']      = to_float(df['FG'], 10)
    out['wind_direction_deg'] = to_float(df['DDVEC'])
    out['temperature_c']      = to_float(df['TG'], 10)
    out['air_pressure_hpa']   = to_float(df['PG'], 10)
    out['sunshine_hours']     = to_float(df['SQ'], 10, clip=True)
    out['precipitation_mm']   = to_float(df['RH'], 10, clip=True)
    out['max_wind_gust_ms']   = to_float(df['FXX'], 10)
    out['min_wind_speed_ms']  = to_float(df['FHN'], 10)
    out['max_wind_speed_ms']  = to_float(df['FHX'], 10)
    out['humidity_pct']       = to_float(df['UG'])
    out['cloud_cover']        = to_float(df['NG'])
    out['global_radiation']   = to_float(df['Q'])

    os.remove(zip_path)
    os.remove(txt_path)
    return out.sort_values('date').reset_index(drop=True)

all_stations = []
for sid, sname in STATIONS.items():
    df = download_and_parse(sid, sname)
    print(f"  {sname}: {len(df)} rows, missing={df.isnull().sum().sum()}")
    all_stations.append(df)

combined = pd.concat(all_stations).reset_index(drop=True)

feature_cols = ['wind_speed_ms', 'wind_direction_deg', 'temperature_c',
                'air_pressure_hpa', 'sunshine_hours', 'precipitation_mm',
                'max_wind_gust_ms', 'min_wind_speed_ms', 'max_wind_speed_ms',
                'humidity_pct', 'cloud_cover', 'global_radiation']

pivot = combined.pivot_table(index='date', columns='station', values=feature_cols)
pivot.columns = [f"{feat}_{stat.lower().replace(' ', '_')}" for feat, stat in pivot.columns]
pivot = pivot.reset_index()

print(f"\nPivot shape: {pivot.shape}")
print(f"Columns: {list(pivot.columns)}")
print(f"Missing values:\n{pivot.isnull().sum()[pivot.isnull().sum() > 0]}")

combined.to_csv('knmi_all_stations.csv', index=False)
pivot.to_csv('knmi_all_stations_wide.csv', index=False)
print("\nSaved: knmi_all_stations.csv and knmi_all_stations_wide.csv")