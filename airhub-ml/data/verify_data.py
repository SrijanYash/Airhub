"""
Script to verify data processing and X_scaled.npy file integrity.
"""
import os
import numpy as np
import pandas as pd
from datetime import datetime

from utils.logger import setup_logger
import config

logger = setup_logger(__name__)

def verify_processed_data():
    """
    Verify that processed data files exist and have the expected format.
    
    Returns:
        bool: True if verification passes, False otherwise
    """
    processed_dir = os.path.join(os.path.dirname(__file__), "datasets")
    x_scaled_path = os.path.join(processed_dir, "X_scaled.npy")
    y_scaled_path = os.path.join(processed_dir, "y_scaled.npy")
    
    # Check if files exist
    if not os.path.exists(x_scaled_path):
        logger.error(f"X_scaled.npy not found at {x_scaled_path}")
        return False
    
    if not os.path.exists(y_scaled_path):
        logger.error(f"y_scaled.npy not found at {y_scaled_path}")
        return False
    
    try:
        # Load the data
        X_scaled = np.load(x_scaled_path)
        y_scaled = np.load(y_scaled_path)
        
        # Verify shape and data type
        logger.info(f"X_scaled shape: {X_scaled.shape}")
        logger.info(f"y_scaled shape: {y_scaled.shape}")
        
        # Check for NaN values
        if np.isnan(X_scaled).any():
            logger.error("X_scaled contains NaN values")
            return False
        
        if np.isnan(y_scaled).any():
            logger.error("y_scaled contains NaN values")
            return False
        
        # Check for infinity values
        if np.isinf(X_scaled).any():
            logger.error("X_scaled contains infinity values")
            return False
        
        if np.isinf(y_scaled).any():
            logger.error("y_scaled contains infinity values")
            return False
        
        # Check expected dimensions
        if len(X_scaled.shape) != 3:
            logger.error(f"X_scaled should be 3-dimensional, but got {len(X_scaled.shape)} dimensions")
            return False
        
        # Verify sequence length matches config
        if X_scaled.shape[1] != config.SEQUENCE_LENGTH:
            logger.error(f"X_scaled sequence length {X_scaled.shape[1]} does not match config {config.SEQUENCE_LENGTH}")
            return False
        
        # Verify feature count matches expected features
        expected_feature_count = len(config.FEATURES)
        if X_scaled.shape[2] != expected_feature_count:
            logger.error(f"X_scaled feature count {X_scaled.shape[2]} does not match expected {expected_feature_count}")
            return False
        
        logger.info("Data verification passed successfully")
        return True
        
    except Exception as e:
        logger.error(f"Error during data verification: {e}")
        return False

def verify_data_directory():
    """
    Verify that the data directory structure is correct and contains necessary files.
    
    Returns:
        bool: True if verification passes, False otherwise
    """
    data_dir = os.path.dirname(__file__)
    datasets_dir = os.path.join(data_dir, "datasets")
    
    # Check if datasets directory exists
    if not os.path.exists(datasets_dir):
        logger.error(f"Datasets directory not found at {datasets_dir}")
        return False
    
    # List all files in the datasets directory
    files = os.listdir(datasets_dir)
    logger.info(f"Files in datasets directory: {files}")
    
    # Check for required files
    required_files = ["X_scaled.npy", "y_scaled.npy"]
    missing_files = [f for f in required_files if f not in files]
    
    if missing_files:
        logger.error(f"Missing required files: {missing_files}")
        return False
    
    logger.info("Data directory verification passed successfully")
    return True

if __name__ == "__main__":
    logger.info("Starting data verification")
    dir_verified = verify_data_directory()
    data_verified = verify_processed_data()
    
    if dir_verified and data_verified:
        logger.info("All verifications passed successfully")
    else:
        logger.error("Verification failed")