# Airhub ML System

## Overview
This system processes air quality and weather data to train machine learning models for prediction.

## Recent Updates
- Migrated OpenAQ API from V2 to V3
- Updated OpenWeather API authentication
- Added data verification and integrity checks
- Implemented robust error handling for API integrations

## Setup

### Prerequisites
- Python 3.7+
- Required packages (install via `pip install -r requirements.txt`)

### Configuration
1. Set your OpenWeather API key:
   - Edit `config.py` directly, or
   - Set the `OPENWEATHER_API_KEY` environment variable

## Usage

### Data Processing
```
python -m data.preprocess
```

### Verify Data Integrity
```
python -m data.verify_data
```

### Run API Tests
```
python -m unittest tests.test_api_integration
```

### Start Training
```
python -m api.train
```

## Troubleshooting
- If you encounter API errors, check your API keys and internet connection
- For data processing errors, run the verification script to check data integrity
- See logs in `airhub.log` for detailed error information

## Documentation
- See `docs/API_CHANGES.md` for detailed information about API updates