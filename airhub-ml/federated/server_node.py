"""
Defines Flower server logic for federated learning.
"""
import flwr as fl
from typing import List, Tuple, Dict, Optional
import numpy as np
import os
import csv
import sys

# Add the project root (airhub-ml) to the system path to resolve module imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model.lstm_model import create_lstm_model, save_model
from utils.logger import setup_logger
import config

logger = setup_logger(__name__)

def weighted_average(metrics: List[Tuple[int, Dict]]) -> Dict:
    """
    Calculate weighted average of metrics.
    
    Args:
        metrics: List of tuples (num_examples, metrics_dict)
        
    Returns:
        Averaged metrics dictionary
    """
    # Calculate weighted average
    weighted_metrics = {}
    total_examples = 0
    
    for num_examples, metrics_dict in metrics:
        total_examples += num_examples
        for key, value in metrics_dict.items():
            if key not in weighted_metrics:
                weighted_metrics[key] = 0
            weighted_metrics[key] += value * num_examples
    
    # Normalize by total number of examples
    for key in weighted_metrics:
        weighted_metrics[key] /= total_examples
    
    return weighted_metrics

def get_fit_config_fn(rounds: int):
    def fit_config(server_round: int) -> Dict:
        logger.info(f"Starting federated round {server_round}/{rounds}")
        return {
            "epochs": config.EPOCHS,
            "batch_size": config.BATCH_SIZE,
        }

    return fit_config

def get_evaluate_fn(input_shape, output_shape, rounds):
    """
    Get evaluation function for server-side evaluation.
    
    Args:
        input_shape: Input shape for the model
        output_shape: Output shape for the model
        
    Returns:
        Evaluation function
    """
    # Load data and take last 20% as test split
    from data.preprocess import load_processed_data
    X_scaled, y_scaled, _, _ = load_processed_data()
    test_count = max(1, int(0.2 * len(X_scaled)))
    X_test, y_test = X_scaled[-test_count:], y_scaled[-test_count:]
    records_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "federated", "records")
    os.makedirs(records_dir, exist_ok=True)
    metrics_path = os.path.join(records_dir, "metrics.csv")
    if not os.path.exists(metrics_path):
        with open(metrics_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["timestamp", "loss", "mae"])
    
    # Define evaluation function
    def evaluate(server_round: int, weights: List[np.ndarray], eval_config: Dict) -> Optional[Tuple[float, Dict]]:
        """
        Evaluate model on test data.
        
        Args:
            server_round: Current federated round
            weights: Model weights as a list of NumPy arrays
            eval_config: Evaluation config from Flower
            
        Returns:
            Tuple of (loss, metrics)
        """
        # Create model
        model = create_lstm_model(input_shape, output_shape)
        
        # Set weights
        model.set_weights(weights)
        save_model(model)
        
        # Evaluate model
        loss, mae = model.evaluate(X_test, y_test, verbose=0)
        try:
            from datetime import datetime
            with open(metrics_path, "a", newline="") as f:
                writer = csv.writer(f)
                writer.writerow([datetime.utcnow().isoformat(), float(loss), float(mae)])
        except Exception as e:
            logger.error(f"Failed to write metrics: {e}")
        if server_round == 0:
            logger.info(f"Initial evaluation complete: loss={float(loss):.4f}, mae={float(mae):.4f}")
        else:
            logger.info(f"Completed federated round {server_round}/{rounds}: loss={float(loss):.4f}, mae={float(mae):.4f}")
        return loss, {"mae": float(mae)}
    
    return evaluate

def start_server(min_clients=config.FL_MIN_CLIENTS,
                 max_clients=config.FL_MAX_CLIENTS,
                 rounds=config.FL_ROUNDS,
                 server_address=config.FL_SERVER_ADDRESS,
                 persistent=False):
    """
    Start a Flower server.

    Args:
        min_clients: Minimum number of clients
        max_clients: Maximum number of clients
        rounds: Number of federated learning rounds
        server_address: Server address
        persistent: If True, server stays running after rounds complete (for hosting)
    """
    # Load data to get input and output shapes
    from data.preprocess import load_processed_data
    X_scaled, y_scaled, _, _ = load_processed_data()
    if X_scaled.size == 0 or X_scaled.ndim != 3:
        logger.warning("Invalid processed data shape; regenerating via preprocess_data")
        from data.preprocess import preprocess_data
        X_scaled, y_scaled, _, _ = preprocess_data(save_to_file=True)

    input_shape = (X_scaled.shape[1], X_scaled.shape[2])
    output_shape = y_scaled.shape[1] if y_scaled.ndim > 1 else 1

    # Define strategy
    strategy = fl.server.strategy.FedAvg(
        fraction_fit=1.0,  # Sample 100% of available clients for training
        fraction_evaluate=1.0,  # Sample 100% of available clients for evaluation
        min_fit_clients=min_clients,  # Minimum number of clients to train
        min_evaluate_clients=min_clients,  # Minimum number of clients to evaluate
        min_available_clients=min_clients,  # Minimum number of available clients
        evaluate_fn=get_evaluate_fn(input_shape, output_shape, rounds),  # Server-side evaluation function
        on_fit_config_fn=get_fit_config_fn(rounds),  # Configuration for client training
        on_evaluate_config_fn=lambda server_round: {},  # Configuration for client evaluation
        accept_failures=False,  # Don't accept failures
    )

    # Start server
    logger.info(f"Starting server at {server_address} with {min_clients}-{max_clients} clients for {rounds} rounds")
    if persistent:
        logger.info("Running in PERSISTENT MODE - server will stay running after rounds complete for continuous training")
        logger.info(f"Server is now LISTENING on {server_address} - clients can connect")

    fl.server.start_server(
        server_address=server_address,
        config=fl.server.ServerConfig(num_rounds=rounds),
        strategy=strategy,
    )

    logger.info(f"Federated training finished after {rounds} rounds. Latest global model saved to {config.MODEL_SAVE_PATH}")

    if persistent:
        import time
        logger.info("Persistent server is now IDLE - waiting for clients to connect for additional training rounds...")
        logger.info("Press Ctrl+C to stop the server")
        try:
            while True:
                time.sleep(60)
        except KeyboardInterrupt:
            logger.info("Persistent server shutting down")

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="AirHub Federated Learning Server")
    parser.add_argument("--persistent", action="store_true", help="Run in persistent mode (stay running after rounds)")
    parser.add_argument("--rounds", type=int, default=config.FL_ROUNDS, help="Number of federated rounds")
    parser.add_argument("--min-clients", type=int, default=config.FL_MIN_CLIENTS, help="Minimum clients")
    args = parser.parse_args()

    start_server(rounds=args.rounds, min_clients=args.min_clients, persistent=args.persistent)
