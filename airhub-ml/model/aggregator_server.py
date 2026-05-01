"""
Federated server for aggregating model weights.
"""
import numpy as np
import os

from model.lstm_model import create_lstm_model, save_model, get_model_weights, set_model_weights
from utils.logger import setup_logger
import config

logger = setup_logger(__name__)

def federated_averaging(weights_list):
    """
    Perform federated averaging on a list of weights.
    
    Args:
        weights_list: List of model weights
        
    Returns:
        Averaged weights
    """
    # Check if weights list is empty
    if not weights_list:
        logger.error("No weights provided for averaging")
        return None
    
    # Initialize with zeros
    avg_weights = [np.zeros_like(w) for w in weights_list[0]]
    
    # Sum all weights
    for weights in weights_list:
        for i, w in enumerate(weights):
            avg_weights[i] += w
    
    # Divide by number of clients
    for i in range(len(avg_weights)):
        avg_weights[i] /= len(weights_list)
    
    return avg_weights

def aggregate_models(model_weights_list, input_shape, output_shape):
    """
    Aggregate models using federated averaging.
    
    Args:
        model_weights_list: List of model weights from clients
        input_shape: Input shape for the model
        output_shape: Output shape for the model
        
    Returns:
        Aggregated model
    """
    logger.info(f"Aggregating {len(model_weights_list)} models")
    
    # Perform federated averaging
    avg_weights = federated_averaging(model_weights_list)
    
    if avg_weights is None:
        logger.error("Failed to aggregate models")
        return None
    
    # Create a new model with the same architecture
    global_model = create_lstm_model(input_shape, output_shape)
    
    # Set the averaged weights
    global_model = set_model_weights(global_model, avg_weights)
    
    # Save the global model
    save_model(global_model, config.MODEL_SAVE_PATH)
    
    logger.info(f"Global model saved to {config.MODEL_SAVE_PATH}")
    
    return global_model

def run_federated_round(client_models, input_shape, output_shape):
    """
    Run a single round of federated learning.
    
    Args:
        client_models: List of client models
        input_shape: Input shape for the model
        output_shape: Output shape for the model
        
    Returns:
        Aggregated global model
    """
    logger.info(f"Running federated round with {len(client_models)} clients")
    
    # Extract weights from client models
    client_weights = [get_model_weights(model) for model in client_models]
    
    # Aggregate models
    global_model = aggregate_models(client_weights, input_shape, output_shape)
    
    return global_model

if __name__ == "__main__":
    # Test federated averaging with dummy models
    from model.train_local import train_local_model
    
    # Train a few local models
    model1, _, _ = train_local_model(city="Delhi", country="IN", epochs=2)
    model2, _, _ = train_local_model(city="Beijing", country="CN", epochs=2)
    model3, _, _ = train_local_model(city="London", country="GB", epochs=2)
    
    # Get input and output shapes
    input_shape = model1.input_shape[1:]
    output_shape = model1.output_shape[1]
    
    # Run federated round
    global_model = run_federated_round([model1, model2, model3], input_shape, output_shape)
    
    print("Federated round complete. Global model created.")