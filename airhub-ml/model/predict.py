"""
Load trained model and predict AQI.
"""
import numpy as np
import pandas as pd
import os
from datetime import datetime, timedelta

import os
import sys

# Add the parent directory to the system path to resolve module imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model.lstm_model import load_saved_model
from utils.logger import setup_logger
from utils.scaler import load_scaler
import config

logger = setup_logger(__name__)

def predict_tomorrow(city=None, country=None):
    """
    Predict tomorrow's AQI, temperature, and weather type.
    
    Args:
        city: City name (default: config.DEFAULT_CITY)
        country: Country code (default: config.DEFAULT_COUNTRY)
        
    Returns:
        Dictionary with predictions
    """
    if city is None:
        city = config.DEFAULT_CITY
    if country is None:
        country = config.DEFAULT_COUNTRY
    
    logger.info(f"Predicting tomorrow's AQI for {city}, {country}")
    
    # Load model
    model = load_saved_model()
    if model is None:
        logger.error("Failed to load model")
        return None
    
    # Load scalers
    models_dir = os.path.dirname(config.MODEL_SAVE_PATH)
    scaler_X_path = os.path.join(models_dir, "scaler_X.pkl")
    scaler_y_path = os.path.join(models_dir, "scaler_y.pkl")
    
    logger.info(f"Loading scalers from {models_dir}")
    scaler_X = load_scaler(scaler_X_path)
    scaler_y = load_scaler(scaler_y_path)
    
    if scaler_X is None or scaler_y is None:
        logger.error(f"Failed to load scalers. X: {scaler_X is not None}, y: {scaler_y is not None}")
        return None
    
    # Load recent data for prediction
    from data.preprocess import preprocess_data
    try:
        X_scaled, _, X, _ = preprocess_data(city, country, save_to_file=False)
        
        if X_scaled is None or X_scaled.size == 0:
            logger.error("No data available for prediction after preprocessing")
            return None
            
        # Ensure input shape matches model expectations
        expected_features = model.input_shape[-1]
        if X_scaled.shape[2] != expected_features:
            logger.warning(f"Feature mismatch: model expects {expected_features}, data has {X_scaled.shape[2]}. Truncating/Padding.")
            if X_scaled.shape[2] > expected_features:
                X_scaled = X_scaled[:, :, :expected_features]
            else:
                # Pad with zeros if fewer features (rare)
                padding = np.zeros((X_scaled.shape[0], X_scaled.shape[1], expected_features - X_scaled.shape[2]))
                X_scaled = np.concatenate([X_scaled, padding], axis=2)
                
        # Use the most recent sequence for prediction
        latest_sequence = X_scaled[-1:]
    except Exception as e:
        logger.error(f"Error during data preprocessing for prediction: {e}")
        return None

    # Predict
    try:
        prediction_scaled = model.predict(latest_sequence, verbose=0)
    except Exception as e:
        logger.error(f"Model prediction failed: {e}")
        return None
    
    # Inverse transform prediction
    prediction = scaler_y.inverse_transform(prediction_scaled)
    
    # Extract predicted values
    aqi = prediction[0, 0]
    temperature = prediction[0, 1] if prediction.shape[1] > 1 else None
    weather_type_idx = int(round(prediction[0, 2])) if prediction.shape[1] > 2 else None
    
    # Map weather type index to name
    weather_type = config.WEATHER_TYPES.get(weather_type_idx, "unknown") if weather_type_idx is not None else None
    
    # Create result dictionary
    tomorrow = datetime.now() + timedelta(days=1)
    
    # Handle NaN values for JSON compliance
    def make_safe_float(val, default=0.0):
        try:
            if val is None or np.isnan(val) or np.isinf(val):
                return default
            return float(val)
        except Exception:
            return default

    safe_aqi = make_safe_float(aqi, 0.0)
    safe_temp = make_safe_float(temperature, 0.0)
    
    result = {
        "city": city,
        "country": country,
        "date": tomorrow.strftime("%Y-%m-%d"),
        "aqi": safe_aqi,
        "aqi_category": _get_aqi_category(safe_aqi),
        "temperature": safe_temp,
        "weather_type": str(weather_type) if weather_type is not None else "Unknown"
    }
    
    # Save prediction to records
    _save_prediction(result)
    
    logger.info(f"Prediction for tomorrow: {result}")
    return result

def _get_aqi_category(aqi):
    """
    Get AQI category based on CPCB (Central Pollution Control Board) standards.

    Args:
        aqi: AQI value

    Returns:
        AQI category string
    """
    if aqi <= 50:
        return "Good"
    elif aqi <= 100:
        return "Satisfactory"
    elif aqi <= 200:
        return "Moderate"
    elif aqi <= 300:
        return "Poor"
    elif aqi <= 400:
        return "Very Poor"
    elif aqi <= 500:
        return "Severe"
    else:
        return "Hazardous"

def _save_prediction(prediction):
    """
    Save prediction to records file.
    
    Args:
        prediction: Prediction dictionary
    """
    # Create records directory if it doesn't exist
    records_dir = os.path.dirname(config.RECORDS_PATH)
    os.makedirs(records_dir, exist_ok=True)
    
    # Create or load records DataFrame
    if os.path.exists(config.RECORDS_PATH):
        records = pd.read_csv(config.RECORDS_PATH)
    else:
        records = pd.DataFrame(columns=[
            "date", "city", "country", "aqi", "aqi_category", 
            "temperature", "weather_type"
        ])
    
    # Add new prediction
    new_record = pd.DataFrame([prediction])
    
    # Check if prediction for this date and city already exists
    mask = (records["date"] == prediction["date"]) & (records["city"] == prediction["city"])
    
    if mask.any():
        # Update existing record
        records.loc[mask] = new_record.values
    else:
        # Append new record
        records = pd.concat([records, new_record], ignore_index=True)
    
    # Save to file
    records.to_csv(config.RECORDS_PATH, index=False)
    logger.info(f"Prediction saved to records: {config.RECORDS_PATH}")

def get_past_predictions(city=None, country=None, days=7):
    """
    Get past predictions from records.
    
    Args:
        city: City name (default: config.DEFAULT_CITY)
        country: Country code (default: config.DEFAULT_COUNTRY)
        days: Number of days to retrieve (default: 7)
        
    Returns:
        DataFrame with past predictions
    """
    if city is None:
        city = config.DEFAULT_CITY
    if country is None:
        country = config.DEFAULT_COUNTRY
    
    # Check if records file exists
    if not os.path.exists(config.RECORDS_PATH):
        logger.warning("No prediction records found")
        return pd.DataFrame()
    
    # Load records
    records = pd.read_csv(config.RECORDS_PATH)
    
    # Filter by city and country - create a copy to avoid SettingWithCopyWarning
    filtered = records[(records["city"] == city) & (records["country"] == country)].copy()
    
    if filtered.empty:
        return filtered

    # Sort by date and get the most recent records
    filtered["date"] = pd.to_datetime(filtered["date"])
    filtered = filtered.sort_values("date", ascending=False).head(days)
    
    # Replace NaN values with None for JSON compliance
    filtered = filtered.where(pd.notnull(filtered), None)
    
    return filtered.sort_values("date")

# Example prediction logic
def make_prediction(input_data):
    model = load_saved_model()
    prediction = model.predict(input_data)
    return prediction

if __name__ == "__main__":
    # Example input data for prediction
    input_data = [[0.5, 0.2, 0.1, 0.7]]  # Replace with actual input data
    result = make_prediction(input_data)
    print("Prediction:", result)
    # Test prediction
    prediction = predict_tomorrow()
    print(f"Prediction: {prediction}")
    
    # Get past predictions
    past = get_past_predictions()
    print(f"Past predictions:\n{past}")