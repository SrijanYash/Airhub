"""
fetch_weather_data.py
---------------------
Fetches 2 years of daily historical weather data from Open-Meteo (free, no API key)
and saves it as a CSV matching the schema:
  date, city, country, temperature, humidity, pressure, wind_speed,
  wind_direction, weather_type, weather_description, day_of_week, month

Usage:
  pip install requests pandas
  python weather_data.py

You can customize the CITIES list below to add/remove locations.
"""
import os
import requests
import pandas as pd
from datetime import date, timedelta

# ── Configuration ──────────────────────────────────────────────────────────────

# Date range: last 2 years up to today
END_DATE   = date.today() - timedelta(days=1)   # yesterday (API lag)
START_DATE = END_DATE - timedelta(days=365 * 2)

OUTPUT_FILE = r"C:\Users\Srijan Yash\Documents\project\Airhub\airhub-ml\data\datasets\sample_weather_data.csv"

# Add or remove cities here. Format: (City, Country, latitude, longitude)
CITIES = [
    # ("Lucknow",    "India",          26.8467,  80.9462),
    ("Delhi",      "India",          28.6139,  77.2090),
    # ("Mumbai",     "India",          19.0760,  72.8777),
    # ("Bangalore",  "India",          12.9716,  77.5946),
    # ("Chennai",    "India",          13.0827,  80.2707),
    # ("Kolkata",    "India",          22.5726,  88.3639),
    # ("London",     "United Kingdom", 51.5074,  -0.1278),
    # ("New York",   "USA",            40.7128, -74.0060),
    # ("Tokyo",      "Japan",          35.6895, 139.6917),
    # ("Sydney",     "Australia",     -33.8688, 151.2093),
]

# ── WMO Weather Code → (type, description) mapping ────────────────────────────
# Source: https://open-meteo.com/en/docs (WMO Weather interpretation codes)
WMO_CODES = {
    0:  ("Clear",        "Clear sky"),
    1:  ("Clear",        "Mainly clear"),
    2:  ("Cloudy",       "Partly cloudy"),
    3:  ("Cloudy",       "Overcast"),
    45: ("Fog",          "Fog"),
    48: ("Fog",          "Depositing rime fog"),
    51: ("Drizzle",      "Light drizzle"),
    53: ("Drizzle",      "Moderate drizzle"),
    55: ("Drizzle",      "Dense drizzle"),
    56: ("Drizzle",      "Light freezing drizzle"),
    57: ("Drizzle",      "Heavy freezing drizzle"),
    61: ("Rain",         "Slight rain"),
    63: ("Rain",         "Moderate rain"),
    65: ("Rain",         "Heavy rain"),
    66: ("Rain",         "Light freezing rain"),
    67: ("Rain",         "Heavy freezing rain"),
    71: ("Snow",         "Slight snowfall"),
    73: ("Snow",         "Moderate snowfall"),
    75: ("Snow",         "Heavy snowfall"),
    77: ("Snow",         "Snow grains"),
    80: ("Rain",         "Slight rain showers"),
    81: ("Rain",         "Moderate rain showers"),
    82: ("Rain",         "Violent rain showers"),
    85: ("Snow",         "Slight snow showers"),
    86: ("Snow",         "Heavy snow showers"),
    95: ("Thunderstorm", "Thunderstorm"),
    96: ("Thunderstorm", "Thunderstorm with slight hail"),
    99: ("Thunderstorm", "Thunderstorm with heavy hail"),
}

DAYS   = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
MONTHS = ["January", "February", "March", "April", "May", "June",
          "July", "August", "September", "October", "November", "December"]

# ── Fetch function ─────────────────────────────────────────────────────────────

def fetch_city(city, country, lat, lon):
    """Fetch daily weather data for one city and return a DataFrame."""
    url = "https://archive-api.open-meteo.com/v1/archive"
    params = {
        "latitude":   lat,
        "longitude":  lon,
        "start_date": START_DATE.isoformat(),
        "end_date":   END_DATE.isoformat(),
        "daily": [
            "temperature_2m_mean",
            "relative_humidity_2m_mean",
            "surface_pressure_mean",
            "wind_speed_10m_mean",
            "wind_direction_10m_dominant",
            "weather_code",
        ],
        "timezone": "auto",
    }

    print(f"  Fetching {city}, {country} ...", end=" ", flush=True)
    resp = requests.get(url, params=params, timeout=30)
    resp.raise_for_status()
    data = resp.json()["daily"]
    print(f"✓ {len(data['time'])} days")

    df = pd.DataFrame({
        "date":            data["time"],
        "temperature":     data["temperature_2m_mean"],
        "humidity":        data["relative_humidity_2m_mean"],
        "pressure":        data["surface_pressure_mean"],
        "wind_speed":      data["wind_speed_10m_mean"],
        "wind_direction":  data["wind_direction_10m_dominant"],
        "weather_code":    data["weather_code"],
    })

    df["date"] = pd.to_datetime(df["date"])
    df["city"]               = city
    df["country"]            = country
    df["weather_type"]       = df["weather_code"].map(lambda c: WMO_CODES.get(c, ("Unknown", "Unknown"))[0])
    df["weather_description"] = df["weather_code"].map(lambda c: WMO_CODES.get(c, ("Unknown", "Unknown"))[1])
    df["day_of_week"]        = df["date"].dt.dayofweek.map(lambda d: DAYS[d])
    df["month"]              = df["date"].dt.month.map(lambda m: MONTHS[m - 1])
    df["date"]               = df["date"].dt.strftime("%Y-%m-%d")

    # Round numeric columns
    for col in ["temperature", "humidity", "pressure", "wind_speed"]:
        df[col] = df[col].round(2)

    return df[[
        "date", "city", "country", "temperature", "humidity", "pressure",
        "wind_speed", "wind_direction", "weather_type", "weather_description",
        "day_of_week", "month"
    ]]

# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    print(f"\n📅 Date range : {START_DATE}  →  {END_DATE}")
    print(f"🌍 Cities     : {len(CITIES)}")
    print(f"📄 Output     : {OUTPUT_FILE}\n")

    all_frames = []
    for city, country, lat, lon in CITIES:
        try:
            df = fetch_city(city, country, lat, lon)
            all_frames.append(df)
        except Exception as e:
            print(f"  ✗ Failed for {city}: {e}")

    if not all_frames:
        print("\n❌ No data fetched. Check your internet connection.")
        return

    combined = pd.concat(all_frames, ignore_index=True)
    combined.to_csv(OUTPUT_FILE, index=False)

    print(f"\n✅ Done! Saved {len(combined):,} rows to '{OUTPUT_FILE}'")
    print(f"\nSample rows:")
    print(combined.head(3).to_string(index=False))

if __name__ == "__main__":
    main()