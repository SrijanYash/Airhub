"""
Evaluation metrics for model performance assessment.
"""
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, accuracy_score
from utils.logger import setup_logger

logger = setup_logger(__name__)

def calculate_rmse(y_true, y_pred):
    """
    Calculate Root Mean Squared Error.
    
    Args:
        y_true: True values
        y_pred: Predicted values
        
    Returns:
        RMSE value
    """
    return np.sqrt(mean_squared_error(y_true, y_pred))

def calculate_mae(y_true, y_pred):
    """
    Calculate Mean Absolute Error.
    
    Args:
        y_true: True values
        y_pred: Predicted values
        
    Returns:
        MAE value
    """
    return mean_absolute_error(y_true, y_pred)

def calculate_accuracy(y_true, y_pred):
    """
    Calculate accuracy for classification tasks.
    
    Args:
        y_true: True class labels
        y_pred: Predicted class labels
        
    Returns:
        Accuracy score
    """
    return accuracy_score(y_true, y_pred)

def evaluate_model(y_true, y_pred, task='regression'):
    """
    Evaluate model performance using appropriate metrics.
    
    Args:
        y_true: True values
        y_pred: Predicted values
        task: Type of task ('regression' or 'classification')
        
    Returns:
        Dictionary of evaluation metrics
    """
    if task == 'regression':
        rmse = calculate_rmse(y_true, y_pred)
        mae = calculate_mae(y_true, y_pred)
        logger.info(f"Model evaluation - RMSE: {rmse:.4f}, MAE: {mae:.4f}")
        return {
            'rmse': rmse,
            'mae': mae
        }
    elif task == 'classification':
        accuracy = calculate_accuracy(y_true, y_pred)
        logger.info(f"Model evaluation - Accuracy: {accuracy:.4f}")
        return {
            'accuracy': accuracy
        }
    else:
        raise ValueError(f"Unknown task type: {task}")