"""
Tests for the model components.
"""
import os
import pytest
import numpy as np
import tensorflow as tf

from model.lstm_model import create_lstm_model, save_model, load_model
from model.predict import predict_tomorrow, categorize_aqi
import config

def test_model_creation():
    """Test model creation."""
    # Create model
    input_shape = (config.SEQUENCE_LENGTH, 5)  # Example input shape
    output_shape = 3  # Example output shape
    model = create_lstm_model(input_shape, output_shape)
    
    # Check model type
    assert isinstance(model, tf.keras.Model)
    
    # Check model layers
    assert len(model.layers) > 0
    assert isinstance(model.layers[0], tf.keras.layers.LSTM)
    
    # Check model input and output shapes
    assert model.input_shape[1:] == input_shape
    assert model.output_shape[1] == output_shape

def test_model_save_load():
    """Test model save and load."""
    # Create temporary model path
    temp_model_path = "temp_test_model.h5"
    
    try:
        # Create model
        input_shape = (config.SEQUENCE_LENGTH, 5)  # Example input shape
        output_shape = 3  # Example output shape
        model = create_lstm_model(input_shape, output_shape)
        
        # Save model
        save_model(model, temp_model_path)
        
        # Check if model file exists
        assert os.path.exists(temp_model_path)
        
        # Load model
        loaded_model = load_model(temp_model_path)
        
        # Check loaded model
        assert isinstance(loaded_model, tf.keras.Model)
        assert loaded_model.input_shape[1:] == input_shape
        assert loaded_model.output_shape[1] == output_shape
    
    finally:
        # Clean up
        if os.path.exists(temp_model_path):
            os.remove(temp_model_path)

def test_aqi_categorization():
    """Test AQI categorization."""
    # Test different AQI values
    assert categorize_aqi(0) == "Good"
    assert categorize_aqi(25) == "Good"
    assert categorize_aqi(50) == "Good"
    assert categorize_aqi(51) == "Moderate"
    assert categorize_aqi(75) == "Moderate"
    assert categorize_aqi(100) == "Moderate"
    assert categorize_aqi(101) == "Unhealthy for Sensitive Groups"
    assert categorize_aqi(150) == "Unhealthy for Sensitive Groups"
    assert categorize_aqi(151) == "Unhealthy"
    assert categorize_aqi(200) == "Unhealthy"
    assert categorize_aqi(201) == "Very Unhealthy"
    assert categorize_aqi(300) == "Very Unhealthy"
    assert categorize_aqi(301) == "Hazardous"
    assert categorize_aqi(500) == "Hazardous"

if __name__ == "__main__":
    # Run tests
    pytest.main(["-xvs", __file__])