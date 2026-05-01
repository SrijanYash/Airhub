"""
Module for preprocessing AQI and weather data.
"""
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os

from data.fetch_aqi_data import fetch_aqi_data
from data.fetch_weather_data import fetch_weather_data
from utils.logger import setup_logger
from utils.scaler import create_scaler, save_scaler, load_scaler
import config

logger = setup_logger(__name__)

def preprocess_data(city=None, country=None, days=None, save_to_file=True):
    """
    Preprocess AQI and weather data for model training.
    
    Args:
        city: City name (default: config.DEFAULT_CITY)
        country: Country code (default: config.DEFAULT_COUNTRY)
        days: Number of days to fetch (default: config.LOOKBACK_DAYS)
        save_to_file: Whether to save processed data to file (default: True)
        
    Returns:
        Tuple of (X, y) for model training
    """
    if city is None:
        city = config.DEFAULT_CITY
    if country is None:
        country = config.DEFAULT_COUNTRY
    if days is None:
        days = config.LOOKBACK_DAYS
    
    logger.info(f"Preprocessing data for {city}, {country}")
    
    # Fetch data
    aqi_data = fetch_aqi_data(city, country, days)
    weather_data = fetch_weather_data(city, country, days, aqi_df=aqi_data)
    
    # Merge datasets on date
    aqi_data['date'] = pd.to_datetime(aqi_data['date'], utc=True, errors='coerce')
    weather_data['date'] = pd.to_datetime(weather_data['date'], utc=True, errors='coerce')

    # Remove duplicate columns from weather_data before merging, except for join keys
    cols_to_use = weather_data.columns.difference(aqi_data.columns).tolist() + ['date', 'city', 'country']
    merged_data = pd.merge(aqi_data, weather_data[cols_to_use], on=['date', 'city', 'country'], how='outer')

    # Ensure date is tz-naive for consistent handling if needed, or keep as UTC
    merged_data['date'] = merged_data['date'].dt.tz_localize(None)

    # Log the size of the merged dataset
    logger.info(f"Merged dataset preview:\n{merged_data.head()}")
    logger.info(f"Merged dataset size: {merged_data.shape}")

    # Check for missing values
    missing_values = merged_data.isnull().sum().sum()
    if missing_values > 0:
        logger.warning(f"Merged dataset contains {missing_values} missing values.")

    # Sort by date
    merged_data = merged_data.sort_values('date')
    
    # Fill missing values
    merged_data = _fill_missing_values(merged_data)
    
    # Ensure time features are numeric for model training
    if 'day_of_week' in merged_data.columns and merged_data['day_of_week'].dtype == 'object':
        day_map = {'Monday': 0, 'Tuesday': 1, 'Wednesday': 2, 'Thursday': 3, 'Friday': 4, 'Saturday': 5, 'Sunday': 6}
        merged_data['day_of_week'] = merged_data['day_of_week'].map(day_map)
    
    if 'month' in merged_data.columns and merged_data['month'].dtype == 'object':
        month_map = {'January': 1, 'February': 2, 'March': 3, 'April': 4, 'May': 5, 'June': 6, 
                     'July': 7, 'August': 8, 'September': 9, 'October': 10, 'November': 11, 'December': 12}
        merged_data['month'] = merged_data['month'].map(month_map)

    # Calculate AQI
    merged_data['aqi'] = _calculate_aqi(merged_data)
    
    # Create sequences for LSTM
    X, y = _create_sequences(merged_data)
    
    # Scale features
    X_scaled, y_scaled, scaler_X, scaler_y = _scale_features(X, y)
    
    # Save scalers only if they were successfully created
    if scaler_X is not None and scaler_y is not None:
        save_scaler(scaler_X, os.path.join(os.path.dirname(config.SCALER_PATH), "scaler_X.pkl"))
        save_scaler(scaler_y, os.path.join(os.path.dirname(config.SCALER_PATH), "scaler_y.pkl"))
    else:
        logger.warning("Scalers were not saved because they are None (likely due to empty input data)")
    
    # Save processed data to file
    if save_to_file:
        processed_dir = os.path.join(os.path.dirname(__file__), "datasets")
        os.makedirs(processed_dir, exist_ok=True)
        
        # Save files
        x_scaled_path = os.path.join(processed_dir, "X_scaled.npy")
        y_scaled_path = os.path.join(processed_dir, "y_scaled.npy")
        x_path = os.path.join(processed_dir, "X.npy")
        y_path = os.path.join(processed_dir, "y.npy")
        
        np.save(x_scaled_path, X_scaled)
        np.save(y_scaled_path, y_scaled)
        np.save(x_path, X)
        np.save(y_path, y)
        
        logger.info(f"Processed data saved to {processed_dir}")
        
        # Verify saved data integrity
        try:
            # Verify X_scaled.npy
            X_scaled_loaded = np.load(x_scaled_path)
            if not np.array_equal(X_scaled, X_scaled_loaded):
                logger.error("X_scaled.npy data integrity check failed")
            else:
                logger.info("X_scaled.npy data integrity verified")
                
            # Check for NaN or infinity values
            if np.isnan(X_scaled_loaded).any() or np.isinf(X_scaled_loaded).any():
                logger.error("X_scaled.npy contains NaN or infinity values")
            
            # Verify expected dimensions
            expected_feature_count = len(config.FEATURES)
            if X_scaled_loaded.shape[2] != expected_feature_count:
                logger.error(f"X_scaled feature count {X_scaled_loaded.shape[2]} does not match expected {expected_feature_count}")
                
        except Exception as e:
            logger.error(f"Error during data verification: {e}")
    
    return X_scaled, y_scaled, X, y

def _fill_missing_values(df):
    """
    Fill missing values in the dataset.
    
    Args:
        df: Input DataFrame
        
    Returns:
        DataFrame with filled missing values
    """
    # Forward fill for missing values
    df = df.fillna(method='ffill')
    
    # If still missing (at the beginning), use backward fill
    df = df.fillna(method='bfill')
    
    # If still missing (empty columns), fill with zeros
    df = df.fillna(0)
    
    return df

def _calculate_aqi(df):
    """
    Calculate Air Quality Index (AQI) from pollutant concentrations using EPA standard method.
    The AQI is calculated for each pollutant and the maximum value is reported as the overall AQI.

    EPA AQI Breakpoints (40 CFR Part 58):
    - PM2.5 (μg/m³, 24hr): 0-12, 12.1-35.4, 35.5-55.4, 55.5-150.4, 150.5-250.4, 250.5-500.4
    - PM10 (μg/m³, 24hr): 0-54, 55-154, 155-254, 255-354, 355-424, 425-604
    - O3 (ppb, 8hr): 0-54, 55-70, 71-85, 86-105, 106-200
    - NO2 (ppb, 1hr): 0-53, 54-100, 101-360, 361-649, 650-1249, 1250-2049
    - SO2 (ppb, 1hr): 0-35, 36-75, 76-185, 186-304, 305-604, 605-1004
    - CO (ppm, 8hr): 0-4.4, 4.5-9.4, 9.5-12.4, 12.5-15.4, 15.5-30.4, 30.5-50.4

    Args:
        df: Input DataFrame with pollutant concentrations

    Returns:
        Series with AQI values
    """
    # EPA AQI breakpoint tables
    # Format: (C_low, C_high, I_low, I_high) for each AQI range

    pm25_breakpoints = [
        (0, 12.0, 0, 50),      # Good
        (12.1, 35.4, 51, 100),  # Moderate
        (35.5, 55.4, 101, 150), # Unhealthy for Sensitive Groups
        (55.5, 150.4, 151, 200), # Unhealthy
        (150.5, 250.4, 201, 300), # Very Unhealthy
        (250.5, 500.4, 301, 500)  # Hazardous
    ]

    pm10_breakpoints = [
        (0, 54, 0, 50),
        (55, 154, 51, 100),
        (155, 254, 51, 150),
        (255, 354, 151, 200),
        (355, 424, 201, 300),
        (425, 604, 301, 500)
    ]

    o3_breakpoints = [
        (0, 54, 0, 50),
        (55, 70, 51, 100),
        (71, 85, 101, 150),
        (86, 105, 151, 200),
        (106, 200, 201, 300)
    ]

    no2_breakpoints = [
        (0, 53, 0, 50),
        (54, 100, 51, 100),
        (101, 360, 101, 150),
        (361, 649, 151, 200),
        (650, 1249, 201, 300),
        (1250, 2049, 301, 500)
    ]

    so2_breakpoints = [
        (0, 35, 0, 50),
        (36, 75, 51, 100),
        (76, 185, 101, 150),
        (186, 304, 151, 200),
        (305, 604, 201, 300),
        (605, 1004, 301, 500)
    ]

    co_breakpoints = [
        (0, 4.4, 0, 50),
        (4.5, 9.4, 51, 100),
        (9.5, 12.4, 101, 150),
        (12.5, 15.4, 151, 200),
        (15.5, 30.4, 201, 300),
        (30.5, 50.4, 301, 500)
    ]

    def get_aqi_subindex(value, breakpoints):
        """Calculate AQI sub-index for a single pollutant value."""
        if pd.isna(value) or value < 0:
            return np.nan

        for c_low, c_high, i_low, i_high in breakpoints:
            if c_low <= value <= c_high:
                # Linear interpolation formula: I = ((I_high - I_low) / (C_high - C_low)) * (C - C_low) + I_low
                return ((i_high - i_low) / (c_high - c_low)) * (value - c_low) + i_low

        # Value exceeds highest breakpoint - extrapolate
        if value > breakpoints[-1][1]:
            last = breakpoints[-1]
            return ((last[3] - last[2]) / (last[1] - last[0])) * (value - last[1]) + last[3]

        return np.nan

    # Calculate AQI sub-indices for each pollutant
    aqi_values = []

    if 'pm25' in df.columns:
        pm25_aqi = df['pm25'].apply(lambda x: get_aqi_subindex(x, pm25_breakpoints))
        aqi_values.append(pm25_aqi)

    if 'pm10' in df.columns:
        pm10_aqi = df['pm10'].apply(lambda x: get_aqi_subindex(x, pm10_breakpoints))
        aqi_values.append(pm10_aqi)

    if 'o3' in df.columns:
        # OpenWeatherMap provides O3 in μg/m³. EPA breakpoints are typically in ppm/ppb.
        # Approx conversion: 1 ppb = 1.96 μg/m³ for O3. 
        # But OWM values can be large, so we divide by 2 to get approximate ppb for EPA breakpoints.
        o3_aqi = (df['o3'] / 2.0).apply(lambda x: get_aqi_subindex(x, o3_breakpoints))
        aqi_values.append(o3_aqi)

    if 'no2' in df.columns:
        # 1 ppb = 1.88 μg/m³ for NO2. Divide by ~1.9 to get ppb.
        no2_aqi = (df['no2'] / 1.9).apply(lambda x: get_aqi_subindex(x, no2_breakpoints))
        aqi_values.append(no2_aqi)

    if 'so2' in df.columns:
        # 1 ppb = 2.62 μg/m³ for SO2. Divide by ~2.6 to get ppb.
        so2_aqi = (df['so2'] / 2.6).apply(lambda x: get_aqi_subindex(x, so2_breakpoints))
        aqi_values.append(so2_aqi)

    if 'co' in df.columns:
        # OpenWeatherMap provides CO in μg/m³. Breakpoints are in mg/m³ (or ppm).
        # Fix: Divide by 1000 to convert μg/m³ to mg/m³ before applying breakpoints.
        co_aqi = (df['co'] / 1000.0).apply(lambda x: get_aqi_subindex(x, co_breakpoints))
        aqi_values.append(co_aqi)

    if not aqi_values:
        logger.warning("No pollutant columns found for AQI calculation")
        return pd.Series([50] * len(df))

    # AQI is the maximum of all sub-indices
    aqi_df = pd.concat(aqi_values, axis=1)
    aqi = aqi_df.max(axis=1)

    # Fill any NaN values with the median
    aqi = aqi.fillna(aqi.median())
    
    # Safety Cap: AQI shouldn't realistically exceed 500-600 in most cases.
    # We'll cap it at 500 to prevent runaway extrapolation from bad input units.
    aqi = aqi.clip(0, 500)

    return aqi

def _create_sequences(df):
    """
    Create sequences for LSTM model.

    Args:
        df: Input DataFrame

    Returns:
        Tuple of (X, y) arrays
    """
    # Select only numeric features
    numeric_features = [f for f in config.FEATURES if f in df.columns and df[f].dtype in ['int64', 'float64', 'int32', 'float32']]

    # Add calculated AQI to features if not already present
    if 'aqi' not in numeric_features:
        numeric_features.append('aqi')

    # Convert to numpy array
    data = df[numeric_features].values
    
    # Check if data has enough rows for sequence creation
    if len(data) <= config.SEQUENCE_LENGTH:
        logger.error(f"Insufficient data for sequence creation: {len(data)} rows available, but SEQUENCE_LENGTH is {config.SEQUENCE_LENGTH}")
        return np.array([]), np.array([])
    
    if data.shape[1] == 0:
        raise ValueError("No features available for sequence creation.")
    # Create sequences
    X, y = [], []
    
    # Pre-calculate target indices to avoid repeated lookups and errors
    feature_index_map = {f: idx for idx, f in enumerate(numeric_features)}
    target_indices = []
    for target in config.TARGETS:
        if target in feature_index_map:
            target_indices.append(feature_index_map[target])
        elif target == 'aqi' and 'aqi' in feature_index_map:
            target_indices.append(feature_index_map['aqi'])
    
    if not target_indices:
        logger.error(f"None of the targets {config.TARGETS} found in numeric features {numeric_features}")
        return np.array([]), np.array([])

    logger.debug(f"Target indices for prediction: {target_indices}")

    for i in range(len(data) - config.SEQUENCE_LENGTH):
        # Input sequence
        X.append(data[i:i+config.SEQUENCE_LENGTH])
        
        # Target values (next step)
        try:
            target_val = data[i+config.SEQUENCE_LENGTH, target_indices]
            y.append(target_val)
        except IndexError as e:
            logger.error(f"IndexError during sequence creation at index {i}: {e}. Data shape: {data.shape}, Target indices: {target_indices}")
            break
    
    return np.array(X), np.array(y)

def _scale_features(X, y):
    """
    Scale features using MinMaxScaler.

    Args:
        X: Input features
        y: Target values

    Returns:
        Tuple of (X_scaled, y_scaled, scaler_X, scaler_y)
    """
    # Check if X or y is empty
    if X.size == 0 or y.size == 0:
        logger.error("Cannot scale features: Input data is empty.")
        return np.array([]), np.array([]), None, None

    try:
        # Reshape X for scaling
        X_reshaped = X.reshape(X.shape[0], -1)

        # Create and fit scalers
        scaler_X = create_scaler(X_reshaped)
        scaler_y = create_scaler(y)

        # Scale features
        X_scaled_reshaped = scaler_X.transform(X_reshaped)
        y_scaled = scaler_y.transform(y)

        # Reshape X back to original shape
        X_scaled = X_scaled_reshaped.reshape(X.shape)

        return X_scaled, y_scaled, scaler_X, scaler_y
    except Exception as e:
        logger.error(f"Error during feature scaling: {e}")
        return np.array([]), np.array([]), None, None

def load_processed_data():
    """
    Load preprocessed data from file.
    
    Returns:
        Tuple of (X_scaled, y_scaled, X, y)
    """
    processed_dir = os.path.join(os.path.dirname(__file__), "datasets")
    
    try:
        X_scaled = np.load(os.path.join(processed_dir, "X_scaled.npy"))
        y_scaled = np.load(os.path.join(processed_dir, "y_scaled.npy"))
        X = np.load(os.path.join(processed_dir, "X.npy"))
        y = np.load(os.path.join(processed_dir, "y.npy"))
        
        logger.info(f"Loaded processed data from {processed_dir}")
        
        if X_scaled.size == 0 or X_scaled.ndim != 3:
            logger.warning(f"Invalid processed X_scaled shape {X_scaled.shape}; regenerating")
            return preprocess_data()
        if y_scaled.size == 0:
            logger.warning("Invalid processed y_scaled; regenerating")
            return preprocess_data()
        
        return X_scaled, y_scaled, X, y
    except Exception as e:
        logger.error(f"Error loading processed data: {e}")
        logger.info("Preprocessing data from scratch")
        return preprocess_data()

if __name__ == "__main__":
    X_scaled, y_scaled, X, y = preprocess_data()
    logger.info(f"X shape: {X.shape}, y shape: {y.shape}")
    logger.info(f"X_scaled shape: {X_scaled.shape}, y_scaled shape: {y_scaled.shape}")