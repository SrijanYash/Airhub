"""
Module for fetching weather data from OpenWeatherMap API.
"""
import requests
import pandas as pd
from datetime import datetime, timedelta
import os
import json
import time
from requests.exceptions import RequestException, HTTPError, Timeout

from utils.logger import setup_logger
import config

logger = setup_logger(__name__)

def fetch_weather_data(city=None, country=None, days=None, forecast=True, aqi_df=None):
    """
    Fetch weather data from Open-Meteo API and save to sample_weather_data_2.csv.
    Uses config values for city, country, and days if not provided.

    Args:
        city: City name (default: config.DEFAULT_CITY)
        country: Country code (default: config.DEFAULT_COUNTRY)
        days: Number of days of historical data (default: 7 for sample data)
        aqi_df: Optional AQI DataFrame (not used)

    Returns:
        DataFrame with weather data saved to sample_weather_data_2.csv
    """
    if city is None:
        city = config.DEFAULT_CITY
    if country is None:
        country = config.DEFAULT_COUNTRY
    if days is None:
        days = 7  # Default to 7 days for sample data

    logger.info(f"Fetching weather data for {city}, {country} for {days} days")

    sample_dir = os.path.join(os.path.dirname(__file__), "datasets")
    os.makedirs(sample_dir, exist_ok=True)
    output_file = os.path.join(sample_dir, "sample_weather_data_2.csv")

    # Get coordinates using config values
    lat = config.DEFAULT_LAT
    lon = config.DEFAULT_LON

    # Try geocoding API first
    try:
        geo_url = "https://geocoding-api.open-meteo.com/v1/search"
        geo_params = {
            "name": city,
            "country": country,
            "count": 1,
            "language": "en",
            "format": "json"
        }
        geo_response = requests.get(geo_url, params=geo_params, timeout=10)
        geo_data = geo_response.json()
        if geo_data and "results" in geo_data:
            lat = geo_data["results"][0]["latitude"]
            lon = geo_data["results"][0]["longitude"]
    except Exception:
        pass  # Use config defaults

    # Calculate date range (last 7 days)
    end_date = datetime.now() - timedelta(days=1)
    start_date = end_date - timedelta(days=days - 1)

    # Build API request using config settings
    params = {
        "latitude": lat,
        "longitude": lon,
        "start_date": start_date.strftime("%Y-%m-%d"),
        "end_date": end_date.strftime("%Y-%m-%d"),
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

    day_names = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    month_names = ['January', 'February', 'March', 'April', 'May', 'June',
                   'July', 'August', 'September', 'October', 'November', 'December']

    try:
        response = requests.get(config.OPENMETEO_API_URL, params=params, timeout=30)
        response.raise_for_status()
        data = response.json()["daily"]

        df = pd.DataFrame({
            "date": data["time"],
            "temperature": data["temperature_2m_mean"],
            "humidity": data["relative_humidity_2m_mean"],
            "pressure": data["surface_pressure_mean"],
            "wind_speed": data["wind_speed_10m_mean"],
            "wind_direction": data["wind_direction_10m_dominant"],
            "weather_code": data["weather_code"],
        })

        df["date"] = pd.to_datetime(df["date"])
        df["city"] = city
        df["country"] = country
        df["weather_type"] = df["weather_code"].map(config.WMO_CODES, na_action='ignore').apply(lambda x: x[0] if x else "Unknown")
        df["weather_description"] = df["weather_code"].map(config.WMO_CODES, na_action='ignore').apply(lambda x: x[1] if x else "Unknown")
        df["day_of_week"] = df["date"].dt.dayofweek.map(lambda d: day_names[d])
        df["month"] = df["date"].dt.month.map(lambda m: month_names[m - 1])
        df["date"] = df["date"].dt.strftime("%Y-%m-%d")

        # Round numeric columns
        for col in ["temperature", "humidity", "pressure", "wind_speed"]:
            df[col] = df[col].round(1)

        # Select and order columns
        df = df[[
            "date", "city", "country", "temperature", "humidity", "pressure",
            "wind_speed", "wind_direction", "weather_type", "weather_description",
            "day_of_week", "month"
        ]]

        df.to_csv(output_file, index=False)
        logger.info(f"Weather data saved to {output_file}")

        return df

    except Exception as e:
        logger.error(f"Failed to fetch weather data: {e}")
        return pd.DataFrame(columns=[
            "date", "city", "country", "temperature", "humidity", "pressure",
            "wind_speed", "wind_direction", "weather_type", "weather_description",
            "day_of_week", "month"
        ])

if __name__ == "__main__":
    fetch_weather_data()