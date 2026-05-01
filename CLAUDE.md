# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

AirHub is a federated learning platform for air quality prediction. It uses LSTM neural networks to predict AQI, temperature, and weather type across multiple locations while preserving data privacy through federated learning.

## Commands

### Setup
```bash
cd airhub-ml
python -m venv venv
venv\Scripts\activate  # Windows
source venv/bin/activate  # Unix
pip install -r requirements.txt
```

### Run API Server
```bash
cd airhub-ml
python main.py
# or
uvicorn main:app --reload
```

### Run Tests
```bash
cd airhub-ml
pytest                    # Run all tests
pytest tests/test_model.py  # Run specific test file
python -m unittest tests.test_api_integration  # API integration tests
```

### Run Federated Learning Simulation
```bash
cd airhub-ml
python -m federated.simulate_training
```

### Start Federated Learning (Server + Clients)

**Option 1: Two terminals (recommended for development)**
```bash
# Terminal 1 - Start persistent server
cd airhub-ml
python -m federated.server_node --persistent

# Terminal 2 - Start client
cd airhub-ml
python -m federated.client_node --city Delhi --country IN --persistent
```

**Option 2: Windows batch script**
```bash
cd airhub-ml
start_federated.bat all
```

**Option 3: Simulation script (all-in-one)**
```bash
cd airhub-ml
python -m federated.simulate_training
```

### Preprocess Data
```bash
cd airhub-ml
python -m data.preprocess
```

### Verify Data Integrity
```bash
cd airhub-ml
python -m data.verify_data
```

## Architecture

### Core Components

**Federated Learning** (`federated/`):
- `server_node.py` - Flower server that aggregates client models using FedAvg strategy
- `client_node.py` - Flower client that trains local LSTM model on city-specific data
- `simulate_training.py` - Orchestrates federated rounds with multiprocessing for parallel client training

**Model** (`model/`):
- `lstm_model.py` - Two-layer LSTM with BatchNormalization and Dropout (64→32→16→output)
- `aggregator_server.py` - Model aggregation logic for federated learning
- `train_local.py` - Local training with train/val split (80/20)
- `predict.py` - Load saved model/scalers, predict tomorrow's values, save to CSV records

**Data Pipeline** (`data/`):
- `fetch_aqi_data.py` - OpenWeatherMap AQI API (v3), falls back to sample data on failure
- `fetch_weather_data.py` - Open-Meteo archive API for historical weather
- `preprocess.py` - Merges AQI + weather data, calculates EPA-standard AQI from pollutants, creates LSTM sequences (7-day lookback), scales with MinMaxScaler, saves to `datasets/*.npy`

**API** (`api/routes/`):
- `predict_route.py` - POST `/api/predict`, GET `/api/predictions/history`
- `train_route.py` - POST `/api/train` (background task), GET `/api/train/status`, GET `/api/flwr/metrics`

### Data Flow

1. **Data Collection**: AQI pollutants (pm25, pm10, o3, no2, so2, co) + weather (temp, humidity, pressure, wind)
2. **AQI Calculation**: EPA standard breakpoints → sub-indices → max = overall AQI
3. **Sequence Creation**: 7-day sequences → predict next day's [aqi, temperature, weather_type]
4. **Scaling**: MinMaxScaler saved as `scaler_X.pkl` and `scaler_y.pkl`
5. **Federated Training**: Clients train locally → server aggregates with FedAvg → global model saved

### Configuration (`config.py`)

Key settings:
- `SEQUENCE_LENGTH = 7` (days)
- `HIDDEN_UNITS = 64`, `BATCH_SIZE = 32`, `EPOCHS = 10`, `LEARNING_RATE = 0.001`
- `FL_MIN_CLIENTS = 1`, `FL_ROUNDS = 10`
- API: `0.0.0.0:8000`, 4 workers
- Environment vars: `OPENWEATHER_API_KEY` (required for real data)

### File Paths

- Model: `model/saved_models/aqi_federated_model.h5`
- Scalers: `model/saved_models/scaler_X.pkl`, `scaler_y.pkl`
- Processed data: `data/datasets/{X,y,X_scaled,y_scaled}.npy`
- Prediction records: `federated/records/predictions.csv`
- FL metrics: `federated/records/metrics.csv`

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/` | API info |
| GET | `/health` | Health check |
| POST | `/api/predict` | Predict tomorrow (body: `{city, country}`) |
| GET | `/api/predictions/history` | Get past predictions |
| POST | `/api/train` | Start federated training (background) |
| GET | `/api/train/status` | Training status |
| GET | `/api/flwr/metrics` | Federated learning metrics |
| POST | `/api/data/ingest` | Ingest custom AQI/weather CSV data |

## Testing Notes

- `test_model.py` - Model creation, save/load, AQI categorization
- `test_api_integration.py` - Mocked API tests for AQI/weather fetchers (error handling, timeouts)
- Tests use sample data fallback when APIs fail

## Common Issues

- **410 Gone from AQI API**: Expected - code falls back to sample data
- **NaN/Inf in scaled data**: Triggers regeneration via `preprocess_data()`
- **Insufficient data for sequences**: Requires >7 days of data (LOOKBACK_DAYS=730)
- **Connection refused on port 8080**: Flower server isn't running. Start with `python -m federated.server_node --persistent`
- **Client timeout waiting for server**: Server address may be wrong or firewall blocking. Default is `127.0.0.1:8080`
