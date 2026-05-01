"""
Module for fetching AQI data from OpenWeatherMap API.
"""
import requests
import pandas as pd
from datetime import datetime, timedelta
import os
import time
from requests.exceptions import RequestException, HTTPError, Timeout

from utils.logger import setup_logger
import config

logger = setup_logger(__name__)

def fetch_aqi_data(city=None, country=None, days=None):
    """
    Fetch AQI data from OpenWeatherMap API for the specified city and time period.

    Args:
        city: City name (default: config.DEFAULT_CITY)
        country: Country code (default: config.DEFAULT_COUNTRY)
        days: Number of days to fetch (default: config.LOOKBACK_DAYS)

    Returns:
        DataFrame with AQI data (hourly)
    """
    if city is None:
        city = config.DEFAULT_CITY
    if country is None:
        country = config.DEFAULT_COUNTRY
    if days is None:
        days = config.LOOKBACK_DAYS

    logger.info(f"Fetching AQI data for {city}, {country}")

    # Check for sample data first (fallback)
    sample_dir = os.path.join(os.path.dirname(__file__), "datasets")
    sample_file = os.path.join(sample_dir, "sample_aqi_data.csv")

    # Check if we have cached API data
    cache_dir = os.path.join(os.path.dirname(__file__), "datasets")
    os.makedirs(cache_dir, exist_ok=True)
    end_date = config.END_DATE
    start_date = config.START_DATE
    cache_file = os.path.join(cache_dir, f"aqi_{city}_{country}_{start_date.strftime('%Y-%m-%d')}_{end_date.strftime('%Y-%m-%d')}.csv")

    # Return cached data if it exists and is recent
    if os.path.exists(cache_file):
        file_age = datetime.now() - datetime.fromtimestamp(os.path.getmtime(cache_file))
        if file_age.total_seconds() < 3600:  # Cache for 1 hour
            logger.info(f"Using cached AQI data from {cache_file}")
            return pd.read_csv(cache_file)

    # Helper function for API requests with retry logic
    def make_api_request(url, params, max_retries=3, retry_delay=2):
        for attempt in range(max_retries):
            try:
                response = requests.get(url, params=params, timeout=10)
                response.raise_for_status()
                return response.json()
            except HTTPError as e:
                if e.response.status_code == 401:
                    logger.error(f"Authentication failed: Invalid API key. Error: {e}")
                    raise
                elif e.response.status_code == 429:
                    logger.warning(f"Rate limit exceeded. Retrying in {retry_delay} seconds...")
                else:
                    logger.warning(f"HTTP error: {e}. Attempt {attempt+1}/{max_retries}")

                if attempt == max_retries - 1:
                    logger.error(f"Failed after {max_retries} attempts: {e}")
                    raise
            except (RequestException, Timeout) as e:
                logger.warning(f"Request failed: {e}. Attempt {attempt+1}/{max_retries}")
                if attempt == max_retries - 1:
                    logger.error(f"Failed after {max_retries} attempts: {e}")
                    raise

            time.sleep(retry_delay)

        return None

    try:
        # Use config coordinates
        lat, lon = config.DEFAULT_LAT, config.DEFAULT_LON

        # Calculate timestamps
        start_ts = config.START_TIME
        end_ts = config.END_TIME

        # Build API request
        params = {
            "lat": lat,
            "lon": lon,
            "start": start_ts,
            "end": end_ts,
            "appid": config.OPENWEATHER_API_KEY,
        }

        logger.info(f"Fetching AQI from OpenWeatherMap: lat={lat}, lon={lon}")

        # Make API request
        data = make_api_request(config.OPENWEATHER_AQI_URL, params)

        if not data or "list" not in data or len(data["list"]) == 0:
            logger.warning(f"No AQI data returned from API for {city}, {country}")
            return _load_sample_aqi_data(city, country)

        # Parse response into hourly records
        records = []
        for item in data["list"]:
            dt = datetime.fromtimestamp(item.get("dt", 0))
            comp = item.get("components", {})

            record = {
                "date": dt.strftime("%Y-%m-%dT%H:%M:%S+00:00"),
                "city": city,
                "country": country,
                "pm25": comp.get("pm2_5", 0),
                "pm10": comp.get("pm10", 0),
                "o3": comp.get("o3", 0),
                "no2": comp.get("no2", 0),
                "so2": comp.get("so2", 0),
                "co": comp.get("co", 0),
                "no": comp.get("no", 0),
                "nh3": comp.get("nh3", 0),
            }
            records.append(record)

        df = pd.DataFrame(records)

        if len(df) > 0:
            # Save to cache
            df.to_csv(cache_file, index=False)
            logger.info(f"AQI data cached: {len(df)} rows")
        else:
            logger.warning("Empty AQI data from API")
            return _load_sample_aqi_data(city, country)

        return df

    except Exception as e:
        logger.error(f"Error fetching AQI data: {e}")
        return _load_sample_aqi_data(city, country)


def _load_sample_aqi_data(city, country):
    """
    Load sample AQI data from the full dataset (17000+ hours).

    Args:
        city: City name
        country: Country code

    Returns:
        DataFrame with sample AQI data
    """
    logger.info("Loading sample AQI data (fallback)")

    sample_dir = os.path.join(os.path.dirname(__file__), "datasets")
    sample_file = os.path.join(sample_dir, "sample_aqi_data.csv")

    if not os.path.exists(sample_file):
        logger.error(f"Sample AQI data file not found: {sample_file}")
        return _generate_fallback_aqi_data(city, country)

    df = pd.read_csv(sample_file)
    logger.info(f"Loaded {len(df)} rows from {sample_file}")

    # Filter by city/country if specified
    if city and country and 'city' in df.columns and 'country' in df.columns:
        filtered = df[(df['city'] == city) & (df['country'] == country)]
        if len(filtered) > 0:
            df = filtered
            logger.info(f"Filtered to {len(df)} rows for {city}, {country}")

    return df


def _generate_fallback_aqi_data(city, country):
    """
    Generate fallback AQI data if sample file is missing.
    """
    logger.info("Generating fallback AQI data")

    sample_dir = os.path.join(os.path.dirname(__file__), "datasets")
    os.makedirs(sample_dir, exist_ok=True)
    sample_file = os.path.join(sample_dir, "sample_aqi_data.csv")

    end_date = datetime.now()
    start_date = end_date - timedelta(days=config.LOOKBACK_DAYS)
    dates = pd.date_range(start=start_date, end=end_date, freq='h')  # Hourly data

    import numpy as np
    np.random.seed(42)

    data = []
    for date in dates:
        hour = date.hour
        day_of_year = date.dayofyear

        # Realistic diurnal and seasonal patterns for Delhi
        seasonal_factor = 1 + 0.3 * np.sin(2 * np.pi * (day_of_year - 15) / 365)  # Winter peaks
        diurnal_factor = 1 + 0.2 * np.sin(2 * np.pi * (hour - 8) / 24)  # Morning/evening peaks

        base_pm25 = 50 * seasonal_factor * diurnal_factor
        base_pm10 = base_pm25 * 1.4

        record = {
            "date": date.strftime("%Y-%m-%dT%H:%M:%S+00:00"),
            "city": city,
            "country": country,
            "pm25": max(5, base_pm25 + np.random.normal(0, 15)),
            "pm10": max(10, base_pm10 + np.random.normal(0, 20)),
            "o3": max(1, 30 + 20 * np.sin(2 * np.pi * hour / 24) + np.random.normal(0, 5)),
            "no2": max(1, 40 + 20 * np.random.normal(0, 1)),
            "so2": max(0.5, 10 + np.random.normal(0, 3)),
            "co": max(0.1, 0.8 + np.random.normal(0, 0.2)),
            "no": max(0.1, 5 + np.random.normal(0, 2)),
            "nh3": max(0.1, 10 + np.random.normal(0, 3)),
        }
        data.append(record)

    df = pd.DataFrame(data)
    df.to_csv(sample_file, index=False)
    logger.info(f"Generated {len(df)} rows of fallback AQI data")

    return df


if __name__ == "__main__":
    aqi_data = fetch_aqi_data()
    print(aqi_data.head())
    print(f"Total rows: {len(aqi_data)}")
