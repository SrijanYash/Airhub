"""
Test cases for API integrations (OpenAQ and OpenWeather).
"""
import unittest
import os
import sys
from unittest.mock import patch, MagicMock
import requests
from requests.exceptions import HTTPError, Timeout

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.fetch_aqi_data import fetch_aqi_data
from data.fetch_weather_data import fetch_weather_data
import config

class TestOpenAQAPI(unittest.TestCase):
    """Test cases for OpenAQ API integration."""
    
    @patch('requests.get')
    def test_successful_aqi_fetch(self, mock_get):
        """Test successful AQI data fetch."""
        # Mock successful response
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "results": [
                {
                    "date": {"local": "2023-01-01T00:00:00"},
                    "parameter": "pm25",
                    "value": 10.5,
                    "unit": "µg/m³",
                    "city": "Delhi",
                    "country": "IN"
                }
            ]
        }
        mock_response.raise_for_status.return_value = None
        mock_get.return_value = mock_response
        
        # Call the function
        result = fetch_aqi_data("Delhi", "IN", 1)
        
        # Verify the result
        self.assertIsNotNone(result)
        self.assertFalse(result.empty)
        
    @patch('requests.get')
    def test_aqi_api_error_handling(self, mock_get):
        """Test error handling for AQI API."""
        # Mock HTTP error
        mock_get.side_effect = HTTPError("410 Client Error: Gone")
        
        # Call the function (should use sample data)
        result = fetch_aqi_data("Delhi", "IN", 1)
        
        # Verify the result (should return sample data)
        self.assertIsNotNone(result)
        self.assertFalse(result.empty)
        
    @patch('requests.get')
    def test_aqi_timeout_handling(self, mock_get):
        """Test timeout handling for AQI API."""
        # Mock timeout
        mock_get.side_effect = Timeout("Request timed out")
        
        # Call the function (should use sample data)
        result = fetch_aqi_data("Delhi", "IN", 1)
        
        # Verify the result (should return sample data)
        self.assertIsNotNone(result)
        self.assertFalse(result.empty)

class TestOpenWeatherAPI(unittest.TestCase):
    """Test cases for OpenWeather API integration."""
    
    @patch('requests.get')
    def test_successful_weather_fetch(self, mock_get):
        """Test successful weather data fetch."""
        # Mock successful responses for both geo and forecast
        mock_geo_response = MagicMock()
        mock_geo_response.json.return_value = [{"lat": 28.6, "lon": 77.2}]
        
        mock_forecast_response = MagicMock()
        mock_forecast_response.json.return_value = {
            "daily": [
                {
                    "dt": 1672531200,  # 2023-01-01
                    "temp": {"day": 25.5},
                    "humidity": 60,
                    "pressure": 1013,
                    "wind_speed": 5.2,
                    "wind_deg": 180,
                    "weather": [{"main": "Clear", "description": "clear sky"}]
                }
            ]
        }
        
        # Configure mock to return different responses for different URLs
        def side_effect(*args, **kwargs):
            url = args[0] if args else kwargs.get('url', '')
            if 'geo' in url:
                return mock_geo_response
            else:
                return mock_forecast_response
                
        mock_get.side_effect = side_effect
        
        # Call the function
        result = fetch_weather_data("Delhi", "IN", 1)
        
        # Verify the result
        self.assertIsNotNone(result)
        self.assertFalse(result.empty)
        
    @patch('requests.get')
    def test_weather_api_error_handling(self, mock_get):
        """Test error handling for weather API."""
        # Mock HTTP error
        mock_get.side_effect = HTTPError("401 Client Error: Unauthorized")
        
        # Call the function (should use sample data)
        result = fetch_weather_data("Delhi", "IN", 1)
        
        # Verify the result (should return sample data)
        self.assertIsNotNone(result)
        self.assertFalse(result.empty)
        
    @patch('requests.get')
    def test_weather_timeout_handling(self, mock_get):
        """Test timeout handling for weather API."""
        # Mock timeout
        mock_get.side_effect = Timeout("Request timed out")
        
        # Call the function (should use sample data)
        result = fetch_weather_data("Delhi", "IN", 1)
        
        # Verify the result (should return sample data)
        self.assertIsNotNone(result)
        self.assertFalse(result.empty)

if __name__ == '__main__':
    unittest.main()