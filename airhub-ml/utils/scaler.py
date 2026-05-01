from sklearn.preprocessing import MinMaxScaler
import os
import pickle
from utils.logger import setup_logger

logger = setup_logger(__name__)

def create_scaler(arr):
    scaler = MinMaxScaler()
    scaler.fit(arr)
    return scaler

def save_scaler(scaler, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(scaler, f)
    logger.info(f"Scaler saved to {path}")

def load_scaler(path):
    try:
        if not os.path.exists(path):
            logger.error(f"Scaler file not found: {path}")
            return None
        with open(path, "rb") as f:
            scaler = pickle.load(f)
            logger.info(f"Scaler loaded from {path}")
            return scaler
    except Exception as e:
        logger.error(f"Error loading scaler from {path}: {e}")
        return None