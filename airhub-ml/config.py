"""
Configuration settings for the AirHub application.
"""
import time
import os
from datetime import datetime, timedelta
from dotenv import load_dotenv

# Load environment variables from .env file if it exists
load_dotenv()

# Default location (can be overridden by user input)
DEFAULT_CITY = "Delhi"
DEFAULT_COUNTRY = "IN"
DEFAULT_LAT = 28.6139
DEFAULT_LON = 77.2090

# Data settings
LOOKBACK_DAYS = 730  # Number of days of historical data to use (~2 years / 17016 hours)
FORECAST_DAYS = 1   # Number of days to forecast (currently only tomorrow)

# Date range for data collection
END_DATE = datetime.now()
START_DATE = END_DATE - timedelta(days=LOOKBACK_DAYS)
START_DATE_STR = START_DATE.strftime("%Y-%m-%d")
END_DATE_STR = END_DATE.strftime("%Y-%m-%d")
end_time = int(time.time())
start_time = end_time - (7 * 24 * 60 * 60)
START_TIME = start_time
END_TIME = end_time

# API Key for AQI data (OpenWeatherMap)
OPENWEATHER_API_KEY = os.getenv("OPENWEATHER_API_KEY")
OPENWEATHER_AQI_URL = "http://api.openweathermap.org/data/2.5/air_pollution/history"

# API for weather data (Open-Meteo)
OPENMETEO_API_URL = "https://archive-api.open-meteo.com/v1/archive"
OPENMETEO_API_PARAMS = {
    "latitude": DEFAULT_LAT,
    "longitude": DEFAULT_LON,
    "start_date": START_DATE_STR,
    "end_date": END_DATE_STR,
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

# Model parameters
SEQUENCE_LENGTH = 7  # Number of days to use as input sequence
HIDDEN_UNITS = 64    # Number of LSTM units
BATCH_SIZE = 32
EPOCHS = 10          # Number of training epochs (increased for larger dataset)
LEARNING_RATE = 0.001

# Federated learning settings
MIN_CLIENTS = 1      # Minimum number of clients required for federated training
MAX_CLIENTS = 10     # Maximum number of clients to simulate
FL_ROUNDS = 10       # Number of federated learning rounds (increased for better convergence)
FL_SERVER_ADDRESS = "127.0.0.1:8080"
FL_MIN_CLIENTS = MIN_CLIENTS
FL_MAX_CLIENTS = MAX_CLIENTS

# API settings
API_HOST = "0.0.0.0"
API_PORT = 8005
API_WORKERS = 4

# File paths
MODEL_SAVE_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "model", "saved_models", "aqi_federated_model.h5"))
RECORDS_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "federated", "records", "predictions.csv"))
SCALER_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "model", "saved_models", "scaler.pkl"))
# Fix: Corrected PREDICTIONS_RECORD_PATH to use a CSV instead of a model file
PREDICTIONS_RECORD_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "federated", "records", "predictions.csv"))

# Features to use for prediction
FEATURES = [
    "aqi", "pm25", "pm10", "o3", "no2", "so2", "co",  # AQI components
    "temperature", "humidity", "pressure", "wind_speed", "wind_direction",  # Weather components
    "day_of_week", "month"  # Time features
]

# Target variables to predict
TARGETS = ["aqi", "temperature", "weather_type"]

# Weather type mapping (for classification)
WEATHER_TYPES = WMO_CODES = {
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
