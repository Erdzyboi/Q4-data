import os
import pandas as pd
from pymongo import MongoClient, ASCENDING
from dotenv import load_dotenv

load_dotenv()

MONGODB_URI = os.getenv("MONGODB_URI")
client = MongoClient(MONGODB_URI)
db = client["wind_energy_q4"]
collection = db["wind_energy_daily"]

# Load CSV
df = pd.read_csv("nl_wind_energy_clean.csv", parse_dates=["date"])

# Feature engineering same as ml_model.py
df["wind_speed_avg"] = df[[
    "wind_speed_ms_de_bilt", "wind_speed_ms_de_kooy",
    "wind_speed_ms_hoek_van_holland", "wind_speed_ms_vlissingen",
    "wind_speed_ms_eelde"
]].mean(axis=1)

df["is_sw_wind"] = df["wind_direction_deg_hoek_van_holland"].between(180, 270).astype(int)

# Convert date to string so MongoDB stores it cleanly
df["date"] = df["date"].dt.strftime("%Y-%m-%d")

# Upload
print(f"Dropping existing collection if any...")
collection.drop()

records = df.to_dict(orient="records")
print(f"Uploading {len(records)} records to MongoDB Atlas...")
collection.insert_many(records)
print(f"Done. {collection.count_documents({})} documents in collection.")

# Index on date
collection.create_index([("date", ASCENDING)], name="date_index")
print("Index created on 'date' field.")

# Index on wind_speed_avg for fast range queries
collection.create_index([("wind_speed_avg", ASCENDING)], name="wind_speed_index")
print("Index created on 'wind_speed_avg' field.")

client.close()
print("\nUpload complete.")
