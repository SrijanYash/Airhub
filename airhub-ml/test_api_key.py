import requests
import json
import time
import csv
import os
from datetime import datetime, timezone

# Configuration for the OpenWeatherMap API

API_KEY = os.getenv("OPENWEATHER_API_KEY")
LATITUDE = 28.6139
LONGITUDE = 77.2090

# Calculate timestamps for the last 6 months (4800 hours)
end_time = int(time.time())
start_time = end_time - (7 * 24 * 60 * 60)

# Construct the API URL for historical air pollution data
url = f"http://api.openweathermap.org/data/2.5/air_pollution/history?lat={LATITUDE}&lon={LONGITUDE}&start={start_time}&end={end_time}&appid={API_KEY}"

try:
    # Make the API request
    response = requests.get(url)
    response.raise_for_status()  # Raise an exception for bad status codes (4xx or 5xx)

    # Parse the JSON response
    data = response.json()

    # Define the path for the output CSV file
    output_file = "c:\\Users\\Srijan Yash\\Documents\\project\\Airhub\\airhub-ml\\data\\datasets\\sample_aqi_data.csv"

    # Open the CSV file for writing
    with open(output_file, "w", newline="") as csvfile:
        # Define the CSV header
        fieldnames = ["date", "city", "country","aqi", "pm25", "pm10", "o3", "no2", "so2", "co", "no", "nh3"]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

        # Write the header row
        writer.writeheader()

        # Process each data point from the API response
        for entry in data.get("list", []):
            # Extract the timestamp and convert it to ISO 8601 format
            dt_object = datetime.fromtimestamp(entry["dt"], tz=timezone.utc)
            iso_date = dt_object.isoformat()

            # Write the data to the CSV file
            writer.writerow({
                "date": iso_date,
                "city": "Delhi",
                "country": "IN",
                "aqi": entry["main"].get("aqi", ""),
                "pm25": entry["components"].get("pm2_5", ""),
                "pm10": entry["components"].get("pm10", ""),
                "o3": entry["components"].get("o3", ""),
                "no2": entry["components"].get("no2", ""),
                "so2": entry["components"].get("so2", ""),
                "co": entry["components"].get("co", ""),
                "no": entry["components"].get("no", ""),
                "nh3": entry["components"].get("nh3", "")
            })

    print(f"Data successfully written to {output_file}")

except requests.exceptions.RequestException as e:
    print(f"Error making API request: {e}")
except json.JSONDecodeError:
    print("Error parsing JSON response. The response may not be valid JSON.")
except Exception as e:
    print(f"An unexpected error occurred: {e}")