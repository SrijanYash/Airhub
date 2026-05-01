"""
Simulate multiple federated learning clients locally.
"""
import os
import threading
import time
import multiprocessing
from typing import List, Tuple

from model.train_local import train_local_model
from model.aggregator_server import run_federated_round
from utils.logger import setup_logger
import config

logger = setup_logger(__name__)

# Sample cities for simulation
SAMPLE_CITIES = [
    ("Delhi", "IN"),
    # ("Beijing", "CN"),
    # ("London", "GB"),
    # ("New York", "US"),
    # ("Tokyo", "JP"),
    # ("Paris", "FR"),
    # ("Mumbai", "IN"),
    # ("Sydney", "AU"),
    # ("Berlin", "DE"),
    # ("Mexico City", "MX")
]

def train_client(city: str, country: str, epochs: int = config.EPOCHS) -> Tuple:
    """
    Train a client model.
    
    Args:
        city: City name
        country: Country code
        epochs: Number of training epochs
        
    Returns:
        Tuple of (model, metrics, data)
    """
    logger.info(f"Training client model for {city}, {country}")
    return train_local_model(city, country, epochs)

def simulate_federated_learning(num_clients: int = 3, 
                               num_rounds: int = config.FL_ROUNDS,
                               epochs_per_round: int = config.EPOCHS) -> None:
    """
    Simulate federated learning with multiple clients.
    
    Args:
        num_clients: Number of clients to simulate
        num_rounds: Number of federated learning rounds
        epochs_per_round: Number of epochs per round
    """
    if num_clients > len(SAMPLE_CITIES):
        logger.warning(f"Requested {num_clients} clients, but only {len(SAMPLE_CITIES)} cities available")
        num_clients = len(SAMPLE_CITIES)
    
    # Select cities for simulation
    selected_cities = SAMPLE_CITIES[:num_clients]
    
    logger.info(f"Starting federated learning simulation with {num_clients} clients for {num_rounds} rounds")
    
    # Create directory for records if it doesn't exist
    os.makedirs(os.path.dirname(config.PREDICTIONS_RECORD_PATH), exist_ok=True)
    
    # Run federated learning rounds
    for round_num in range(1, num_rounds + 1):
        logger.info(f"Starting round {round_num}/{num_rounds}")
        
        # Train client models in parallel
        client_models = []
        client_data = []
        
        with multiprocessing.Pool(processes=min(num_clients, multiprocessing.cpu_count())) as pool:
            results = pool.starmap(
                train_client,
                [(city, country, epochs_per_round) for city, country in selected_cities]
            )
        
        # Extract models and data
        for model, metrics, data in results:
            client_models.append(model)
            client_data.append(data)
            logger.info(f"Client metrics: {metrics}")
        
        # Get input and output shapes from first model
        input_shape = client_models[0].input_shape[1:]
        output_shape = client_models[0].output_shape[1]
        
        # Run federated round
        global_model = run_federated_round(client_models, input_shape, output_shape)
        
        logger.info(f"Completed round {round_num}/{num_rounds}")
    
    logger.info("Federated learning simulation complete")

def wait_for_server(server_address: str, timeout: int = 30) -> bool:
    """
    Wait for the Flower server to become available.

    Args:
        server_address: Server address in format "host:port"
        timeout: Maximum time to wait in seconds

    Returns:
        True if server is reachable, False if timeout exceeded
    """
    import socket
    from datetime import datetime, timedelta

    host, port = server_address.split(":")
    port = int(port)
    deadline = datetime.now() + timedelta(seconds=timeout)
    poll_interval = 1.0

    while datetime.now() < deadline:
        try:
            sock = socket.create_connection((host, port), timeout=2)
            sock.close()
            return True
        except (ConnectionRefusedError, socket.timeout, OSError):
            time.sleep(poll_interval)

    return False


def simulate_with_real_server():
    """
    Simulate federated learning with a real Flower server.
    This function starts a server and multiple clients in separate processes.
    """
    from federated.server_node import start_server
    from federated.client_node import start_client

    # Start server in a separate process
    server_process = multiprocessing.Process(
        target=start_server,
        kwargs={
            "min_clients": config.FL_MIN_CLIENTS,
            "max_clients": config.FL_MAX_CLIENTS,
            "rounds": config.FL_ROUNDS,
            "server_address": config.FL_SERVER_ADDRESS
        }
    )
    server_process.start()

    # Wait for server to start (with proper connection check)
    logger.info(f"Waiting for server to start at {config.FL_SERVER_ADDRESS}...")
    if not wait_for_server(config.FL_SERVER_ADDRESS, timeout=30):
        logger.error("Server failed to start within timeout")
        server_process.terminate()
        return

    logger.info("Server is ready, starting clients...")

    # Start clients in separate processes
    client_processes = []
    for i in range(config.FL_MIN_CLIENTS):
        city, country = SAMPLE_CITIES[i]
        client_process = multiprocessing.Process(
            target=start_client,
            kwargs={
                "city": city,
                "country": country,
                "server_address": config.FL_SERVER_ADDRESS
            }
        )
        client_processes.append(client_process)
        client_process.start()

    # Wait for clients to complete
    for client_process in client_processes:
        client_process.join()

    # Terminate server
    server_process.terminate()

    logger.info("Federated learning with real server complete")

if __name__ == "__main__":
    # Run simulation
    simulate_federated_learning(num_clients=3, num_rounds=2, epochs_per_round=2)