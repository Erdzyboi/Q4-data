import requests
import pandas as pd
import time

with open('.env') as f:
    for line in f:
        if line.startswith('NED_API_KEY'):
            api_key = line.strip().split('=')[1]
            break

print(f"API key loaded: {api_key[:8]}...")

BASE_URL = "https://api.ned.nl/v1/utilizations"
HEADERS = {"X-AUTH-TOKEN": api_key, "Accept": "application/json"}

def fetch_ned_year(type_id, year, label):
    all_items = []
    page = 1
    while True:
        params = {
            "point": 0,
            "type": type_id,
            "granularity": 6,
            "granularitytimezone": 1,
            "classification": 2,
            "activity": 1,
            "validfrom[after]": f"{year}-01-01",
            "validfrom[strictly_before]": f"{year+1}-01-01",
            "itemsPerPage": 200,
            "page": page
        }
        r = requests.get(BASE_URL, headers=HEADERS, params=params, timeout=30)
        if r.status_code != 200:
            print(f"  Error {r.status_code}: {r.text[:200]}")
            return pd.DataFrame()
        data = r.json()
        items = data if isinstance(data, list) else data.get("hydra:member", [])
        if not items:
            break
        all_items.extend(items)
        print(f"  {label} {year} page {page}: {len(items)} rows")
        if len(items) < 200:
            break
        page += 1
        time.sleep(0.3)

    if not all_items:
        return pd.DataFrame()

    df = pd.DataFrame(all_items)
    df["date"] = pd.to_datetime(df["validfrom"]).dt.date
    df["volume_kwh"] = pd.to_numeric(df["volume"], errors="coerce")
    daily = df.groupby("date")["volume_kwh"].sum().reset_index()
    daily.columns = ["date", label]
    return daily

all_onshore, all_offshore = [], []

for year in [2021, 2022, 2023]:
    print(f"\nFetching wind onshore {year}...")
    df = fetch_ned_year(1, year, "wind_onshore_kwh")
    if not df.empty:
        all_onshore.append(df)
    print(f"Fetching wind offshore {year}...")
    df = fetch_ned_year(17, year, "wind_offshore_kwh")
    if not df.empty:
        all_offshore.append(df)

onshore  = pd.concat(all_onshore).reset_index(drop=True)
offshore = pd.concat(all_offshore).reset_index(drop=True)
merged   = onshore.merge(offshore, on="date", how="outer")
merged["date"] = pd.to_datetime(merged["date"])
merged = merged.sort_values("date").reset_index(drop=True)

merged["wind_onshore_gwh"]  = merged["wind_onshore_kwh"]  / 1e6
merged["wind_offshore_gwh"] = merged["wind_offshore_kwh"] / 1e6
merged["wind_energy_gwh"]   = merged["wind_onshore_gwh"] + merged["wind_offshore_gwh"]

result = merged[["date","wind_onshore_gwh","wind_offshore_gwh","wind_energy_gwh"]]

print(f"\nRows: {len(result)}")
print(f"Date range: {result.date.min()} to {result.date.max()}")
print(f"Missing: {result.isnull().sum().sum()}")
print(f"\nSample:\n{result.head(5).to_string()}")
print(f"\nStats:\n{result.describe().round(2).to_string()}")

result.to_csv("ned_wind_daily.csv", index=False)
print("\nSaved: ned_wind_daily.csv")