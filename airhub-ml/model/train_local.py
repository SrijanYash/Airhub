"""
Client-side training for federated learning.
"""
import tensorflow as tf
import numpy as np
import os
from sklearn.model_selection import train_test_split

import os
import sys

# Add the parent directory to the system path to resolve module imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model.lstm_model import create_lstm_model, save_model
from data.preprocess import preprocess_data, load_processed_data
from utils.logger import setup_logger
from utils.evaluation import evaluate_model
import config

logger = setup_logger(__name__)

def train_local_model(city=None, country=None, epochs=None, batch_size=None):
    """
    Train a local model on client data.
    
    Args:
        city: City name (default: config.DEFAULT_CITY)
        country: Country code (default: config.DEFAULT_COUNTRY)
        epochs: Number of training epochs (default: config.EPOCHS)
        batch_size: Batch size for training (default: config.BATCH_SIZE)
        
    Returns:
        Trained model and evaluation metrics
    """
    if city is None:
        city = config.DEFAULT_CITY
    if country is None:
        country = config.DEFAULT_COUNTRY
    if epochs is None:
        epochs = config.EPOCHS
    if batch_size is None:
        batch_size = config.BATCH_SIZE
    
    logger.info(f"Training local model for {city}, {country}")
    logger.info("Logger is initialized and writing logs.")
    
    # Load or preprocess data
    try:
        X_scaled, y_scaled, X, y = load_processed_data()
        logger.info("Loaded preprocessed data")
    except Exception as e:
        logger.warning(f"Could not load preprocessed data: {e}")
        logger.info("Preprocessing data from scratch")
        X_scaled, y_scaled, X, y = preprocess_data(city, country)
    
    # Split data into train and validation sets
    logger.info(f"Dataset X shape: {X.shape if X is not None else 'None'}")
    logger.info(f"Dataset y shape: {y.shape if y is not None else 'None'}")
    logger.info(f"Dataset X preview:\n{X[:5] if X is not None else 'None'}")
    logger.info(f"Dataset y preview:\n{y[:5] if y is not None else 'None'}")
    X_train, X_val, y_train, y_val = train_test_split(
        X_scaled, y_scaled, test_size=0.2, random_state=42
    )
    
    # Create model
    input_shape = (X_train.shape[1], X_train.shape[2])
    output_shape = y_train.shape[1]
    
    model = create_lstm_model(input_shape, output_shape)
    
    # Train model
    logger.info(f"Training model for {epochs} epochs with batch size {batch_size}")
    
    history = model.fit(
        X_train, y_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_data=(X_val, y_val),
        verbose=1
    )
    
    # Evaluate model
    y_pred = model.predict(X_val)
    metrics = evaluate_model(y_val, y_pred, task='regression')
    
    # Save model locally
    local_model_path = os.path.join(
        os.path.dirname(config.MODEL_SAVE_PATH),
        f"local_model_{city}_{country}.h5"
    )
    save_model(model, local_model_path)
    
    return model, metrics, history

# Example training logic
def train_model():
    # Preprocess data
    X_train, y_train = preprocess_data()

    # Create and train the model
    model = create_lstm_model()
    model.fit(X_train, y_train, epochs=10, batch_size=32)

    # Save the trained model
    save_model(model)

if __name__ == "__main__":
    # Test local training
    model, metrics, history = train_local_model()
    print(f"Training complete. Metrics: {metrics}")