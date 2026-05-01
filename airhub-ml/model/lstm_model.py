"""
LSTM model architecture for AQI prediction.
"""
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
import os
import numpy as np

from utils.logger import setup_logger
import config

logger = setup_logger(__name__)

def create_lstm_model(input_shape, output_shape):
    """
    Create LSTM model for AQI prediction.
    
    Args:
        input_shape: Shape of input data (sequence_length, features)
        output_shape: Number of output features
        
    Returns:
        Compiled Keras model
    """
    logger.info(f"Creating LSTM model with input shape {input_shape} and output shape {output_shape}")
    
    model = Sequential([
        LSTM(config.HIDDEN_UNITS, return_sequences=True, input_shape=input_shape),
        BatchNormalization(),
        Dropout(0.2),
        
        LSTM(config.HIDDEN_UNITS // 2, return_sequences=False),
        BatchNormalization(),
        Dropout(0.2),
        
        Dense(config.HIDDEN_UNITS // 4, activation='relu'),
        BatchNormalization(),
        Dropout(0.2),
        
        Dense(output_shape)
    ])
    
    # Compile model
    model.compile(
        optimizer=Adam(learning_rate=config.LEARNING_RATE),
        loss='mse',
        metrics=['mae']
    )
    
    logger.info(f"Model created with {model.count_params()} parameters")
    return model

def save_model(model, path=None):
    """
    Save model to disk.
    
    Args:
        model: Keras model to save
        path: Path to save the model (default: config.MODEL_SAVE_PATH)
    """
    if path is None:
        path = config.MODEL_SAVE_PATH
    
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(path), exist_ok=True)
    
    model.save(path)
    logger.info(f"Model saved to {path}")

def load_saved_model(path=None):
    """
    Load model from disk.

    Args:
        path: Path to load the model from (default: config.MODEL_SAVE_PATH)

    Returns:
        Loaded Keras model
    """
    if path is None:
        path = config.MODEL_SAVE_PATH

    if not os.path.exists(path):
        logger.error(f"Model file not found at {path}")
        return None

    try:
        # Try standard loading first
        model = load_model(path, compile=False)
        logger.info(f"Model loaded from {path}")
        return model
    except Exception as e:
        logger.warning(f"Standard load_model failed: {e}")
        try:
            # Fallback: Recreate architecture and load weights
            # This is more robust against Keras version mismatches
            from data.preprocess import load_processed_data
            X_scaled, y_scaled, _, _ = load_processed_data()
            input_shape = (X_scaled.shape[1], X_scaled.shape[2])
            output_shape = y_scaled.shape[1] if y_scaled.ndim > 1 else 1
            
            model = create_lstm_model(input_shape, output_shape)
            model.load_weights(path)
            logger.info(f"Model weights loaded successfully into fresh architecture from {path}")
            return model
        except Exception as e2:
            logger.error(f"Failed all model loading attempts: {e2}")
            return None

def get_model_weights(model):
    """
    Get model weights as a list of NumPy arrays.
    
    Args:
        model: Keras model
        
    Returns:
        List of NumPy arrays
    """
    return [w.numpy() for w in model.weights]

def set_model_weights(model, weights):
    """
    Set model weights from a list of NumPy arrays.
    
    Args:
        model: Keras model
        weights: List of NumPy arrays
        
    Returns:
        Updated model
    """
    model.set_weights(weights)
    return model

if __name__ == "__main__":
    # Test the model creation
    from data.preprocess import load_processed_data
    
    X_scaled, y_scaled, _, _ = load_processed_data()
    
    input_shape = (X_scaled.shape[1], X_scaled.shape[2])
    output_shape = y_scaled.shape[1]
    
    model = create_lstm_model(input_shape, output_shape)
    model.summary()