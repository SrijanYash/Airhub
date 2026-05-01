"""
Defines Flower client logic for federated learning.
"""
import numpy as np
import flwr as fl
import tensorflow as tf
import os
import sys

# Add the project root (airhub-ml) to the system path to resolve module imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model.lstm_model import create_lstm_model, get_model_weights, set_model_weights
from model.train_local import train_local_model
from utils.logger import setup_logger
import config as app_config

logger = setup_logger(__name__)

class AirHubClient(fl.client.NumPyClient):
    """Flower client for AirHub federated learning."""
    
    def __init__(self, city, country, input_shape, output_shape):
        """
        Initialize AirHub client.
        
        Args:
            city: City name
            country: Country code
            input_shape: Input shape for the model
            output_shape: Output shape for the model
        """
        self.city = city
        self.country = country
        self.input_shape = input_shape
        self.output_shape = output_shape
        
        # Create local model
        self.model = create_lstm_model(input_shape, output_shape)
        
        logger.info(f"Initialized AirHub client for {city}, {country}")
    
    def get_parameters(self, config):
        """
        Get model parameters.
        
        Args:
            config: Configuration dictionary
            
        Returns:
            Model parameters as a list of NumPy arrays
        """
        return get_model_weights(self.model)
    
    def fit(self, parameters, config):
        """
        Train model on local data.
        
        Args:
            parameters: Model parameters from server
            config: Configuration dictionary
            
        Returns:
            Updated model parameters, number of examples, and metrics
        """
        # Set model parameters
        self.model = set_model_weights(self.model, parameters)
        
        # Train model on local data
        epochs = config.get("epochs", app_config.EPOCHS)
        batch_size = config.get("batch_size", app_config.BATCH_SIZE)
        
        logger.info(f"Training local model for {self.city}, {self.country}")
        self.model, metrics, _ = train_local_model(
            self.city, self.country, epochs, batch_size
        )
        
        # Return updated model parameters and metrics
        return get_model_weights(self.model), 1, {"rmse": metrics["rmse"]}
    
    def evaluate(self, parameters, config):
        """
        Evaluate model on local data.
        
        Args:
            parameters: Model parameters from server
            config: Configuration dictionary
            
        Returns:
            Loss, number of examples, and metrics
        """
        # Set model parameters
        self.model = set_model_weights(self.model, parameters)
        
        from data.preprocess import load_processed_data
        X_scaled, y_scaled, _, _ = load_processed_data()
        loss, mae = self.model.evaluate(X_scaled, y_scaled, verbose=0)
        return loss, len(y_scaled), {"mae": float(mae)}

def wait_for_server(server_address: str, timeout: int = 60, poll_interval: float = 2.0) -> bool:
    """
    Wait for the Flower server to become available.

    Args:
        server_address: Server address in format "host:port"
        timeout: Maximum time to wait in seconds
        poll_interval: Time between connection attempts

    Returns:
        True if server is reachable, False if timeout exceeded
    """
    import socket
    from datetime import datetime, timedelta

    host, port = server_address.split(":")
    port = int(port)
    deadline = datetime.now() + timedelta(seconds=timeout)

    logger.info(f"Waiting for server at {server_address} (timeout: {timeout}s)...")

    while datetime.now() < deadline:
        try:
            sock = socket.create_connection((host, port), timeout=2)
            sock.close()
            logger.info(f"Server is reachable at {server_address}")
            return True
        except (ConnectionRefusedError, socket.timeout, OSError):
            logger.debug(f"Server not ready, retrying in {poll_interval}s...")
            time.sleep(poll_interval)

    logger.error(f"Server at {server_address} did not become reachable within {timeout}s")
    return False


def start_client(city, country, server_address=None, wait_for_server_flag: bool = True, wait_timeout: int = 60):
    """
    Start a Flower client.

    Args:
        city: City name
        country: Country code
        server_address: Server address (default: config.FL_SERVER_ADDRESS)
        wait_for_server_flag: If True, wait for server to be available before connecting
        wait_timeout: Maximum time to wait for server in seconds
    """
    if server_address is None:
        server_address = app_config.FL_SERVER_ADDRESS

    from data.preprocess import load_processed_data
    X_scaled, y_scaled, _, _ = load_processed_data()

    input_shape = (X_scaled.shape[1], X_scaled.shape[2])
    output_shape = y_scaled.shape[1]

    client = AirHubClient(city, country, input_shape, output_shape)

    if wait_for_server_flag:
        if not wait_for_server(server_address, timeout=wait_timeout):
            logger.error(
                f"\n{'='*60}\n"
                f"SERVER NOT FOUND at {server_address}\n"
                f"{'='*60}\n"
                f"To start the server, run:\n"
                f"  python -m federated.server_node --persistent\n"
                f"{'='*60}\n"
            )
            raise ConnectionError(f"Cannot connect to server at {server_address}")

    logger.info(f"Starting client for {city}, {country} connecting to {server_address}")
    fl.client.start_numpy_client(server_address=server_address, client=client)

def request_prediction(city: str = None, country: str = None, api_host: str = "http://localhost:8005") -> dict:
    """
    Request AQI prediction from the hosted API server.

    Args:
        city: City name
        country: Country code
        api_host: API server host

    Returns:
        Prediction response dictionary
    """
    import requests

    city = city or app_config.DEFAULT_CITY
    country = country or app_config.DEFAULT_COUNTRY

    url = f"{api_host}/api/predict"
    payload = {"city": city, "country": country}

    logger.info(f"Requesting prediction for {city}, {country} from {url}")

    try:
        response = requests.post(url, json=payload, timeout=30)
        response.raise_for_status()
        result = response.json()
        logger.info(f"Prediction received: AQI={result.get('aqi')}, Temp={result.get('temperature')}")
        return result
    except requests.exceptions.ConnectionError:
        logger.error(f"Cannot connect to API server at {api_host}")
        raise
    except requests.exceptions.Timeout:
        logger.error("Request timed out")
        raise
    except Exception as e:
        logger.error(f"Prediction request failed: {e}")
        raise


if __name__ == "__main__":
    import sys
    import time

    # Parse command line arguments
    city = app_config.DEFAULT_CITY
    country = app_config.DEFAULT_COUNTRY
    persistent = False
    predict_mode = False
    api_host = "http://localhost:8005"
    wait_timeout = 120  # Default 2 minutes wait for server

    for i, arg in enumerate(sys.argv):
        if arg == "--city" and i + 1 < len(sys.argv):
            city = sys.argv[i + 1]
        elif arg == "--country" and i + 1 < len(sys.argv):
            country = sys.argv[i + 1]
        elif arg == "--persistent":
            persistent = True
        elif arg == "--predict":
            predict_mode = True
        elif arg == "--api-host" and i + 1 < len(sys.argv):
            api_host = sys.argv[i + 1]
        elif arg == "--wait-timeout" and i + 1 < len(sys.argv):
            wait_timeout = int(sys.argv[i + 1])

    if predict_mode:
        # Request prediction from API
        try:
            result = request_prediction(city, country, api_host)
            print(f"\n=== Prediction for {city}, {country} ===")
            print(f"Date: {result.get('date', 'N/A')}")
            print(f"AQI: {result.get('aqi', 'N/A')} ({result.get('aqi_category', 'N/A')})")
            print(f"Temperature: {result.get('temperature', 'N/A')}°C")
            print(f"Weather: {result.get('weather_type', 'N/A')}")
            print("=" * 40)
        except Exception as e:
            logger.error(f"Failed to get prediction: {e}")
            sys.exit(1)
    elif persistent:
        logger.info(f"Starting PERSISTENT client for {city}, {country} - will auto-reconnect if server restarts")
        logger.info("Press Ctrl+C to stop the client")
        while True:
            try:
                start_client(city, country, wait_for_server_flag=True, wait_timeout=wait_timeout)
                logger.info("Client connection ended, reconnecting in 10 seconds...")
                time.sleep(10)
            except KeyboardInterrupt:
                logger.info("Persistent client shutting down")
                break
            except ConnectionError as e:
                logger.error(f"{e}")
                logger.info("Waiting 30 seconds before retry...")
                time.sleep(30)
            except Exception as e:
                logger.error(f"Client error: {e}, reconnecting in 10 seconds...")
                time.sleep(10)
    else:
        start_client(city, country, wait_for_server_flag=True, wait_timeout=wait_timeout)