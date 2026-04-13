# AirHub: Federated Learning Platform for Air Quality Prediction

AirHub is a federated learning platform built in Python that predicts tomorrow's air quality, temperature, and weather type (rainy, sunny, cloudy, windy, stormy) using historical data from multiple locations.

## Project Overview

AirHub uses a federated learning approach where multiple clients (representing different locations) train local models on their data, and a central server aggregates these models to create a global model. This approach preserves data privacy while leveraging the collective knowledge from multiple data sources.

### Key Features

- **Federated Learning**: Train models across multiple locations without sharing raw data
- **Air Quality Prediction**: Forecast tomorrow's Air Quality Index (AQI)
- **Temperature Prediction**: Predict tomorrow's temperature
- **Weather Type Classification**: Classify tomorrow's weather type
- **FastAPI Integration**: Serve predictions through a RESTful API
- **Historical Tracking**: Store and retrieve past predictions

## Project Structure

```
airhub-ml/
├── api/                  # API components
│   ├── dependencies.py   # API dependencies
│   └── routes/           # API routes
├── config.py             # Configuration settings
├── data/                 # Data components
│   ├── fetch_aqi_data.py # AQI data fetching
│   ├── fetch_weather_data.py # Weather data fetching
│   └── preprocess.py     # Data preprocessing
├── federated/            # Federated learning components
│   ├── client_node.py    # Federated client
│   ├── server_node.py    # Federated server
│   └── simulate_training.py # Simulation utilities
├── main.py               # Main application entry point
├── model/                # Model components
│   ├── aggregator_server.py # Model aggregation
│   ├── lstm_model.py     # LSTM model definition
│   ├── predict.py        # Prediction utilities
│   └── train_local.py    # Local training
├── requirements.txt      # Project dependencies
├── tests/                # Test suite
│   ├── test_api.py       # API tests
│   ├── test_data_pipeline.py # Data pipeline tests
│   └── test_model.py     # Model tests
└── utils/                # Utility functions
    ├── evaluation.py     # Model evaluation
    ├── logger.py         # Logging utilities
    └── scaler.py         # Data scaling
```

## Installation

1. Clone the repository:
   ```
   git clone https://github.com/yourusername/airhub.git
   cd airhub
   ```

2. Create a virtual environment:
   ```
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

4. Set up environment variables:
   Create a `.env` file in the project root with the following variables:
   ```
   OPENAQ_API_KEY=your_openaq_api_key
   OPENWEATHER_API_KEY=your_openweather_api_key
   ```

## Usage

### Running the API

Start the FastAPI server:

```
cd airhub-ml
uvicorn main:app --reload
```

The API will be available at http://localhost:8000.

### API Endpoints

- `GET /`: Root endpoint
- `GET /health`: Health check endpoint
- `POST /predict`: Predict tomorrow's air quality, temperature, and weather type
- `GET /predictions/history`: Get past predictions
- `POST /train`: Start federated learning training
- `GET /train/status`: Get training status

### Example: Making a Prediction

```python
import requests

response = requests.post(
    "http://localhost:8000/predict",
    json={"city": "Delhi", "country": "IN"}
)

print(response.json())
```

### Running Federated Learning Simulation

```python
from federated.simulate_training import simulate_federated_learning

# Run simulation with 3 clients, 5 rounds, and 10 epochs per round
simulate_federated_learning(num_clients=3, num_rounds=5, epochs_per_round=10)
```

## Testing

Run the test suite:

```
cd airhub-ml
pytest
```

## Data Sources

- Air Quality Data: [OpenAQ API](https://openaq.org/)
- Weather Data: [OpenWeatherMap API](https://openweathermap.org/)

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgements

- [Flower](https://flower.dev/) - Federated Learning Framework
- [TensorFlow](https://www.tensorflow.org/) - Machine Learning Framework
- [FastAPI](https://fastapi.tiangolo.com/) - Web Framework