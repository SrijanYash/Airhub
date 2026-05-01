"""
Logging setup for the AirHub application.
"""
import logging
import sys
from logging.handlers import RotatingFileHandler
import os

def setup_logger(name, log_level=logging.INFO):
    """
    Set up and configure logger for the application.
    
    Args:
        name: Name of the logger
        log_level: Logging level
        
    Returns:
        Configured logger instance
    """
    # Create logger
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    
    # Create formatters
    file_formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    console_formatter = logging.Formatter(
        '%(levelname)s - %(message)s'
    )
    
    # Create console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(console_formatter)
    console_handler.setLevel(log_level)
    
    # Create file handler
    log_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "logs")
    os.makedirs(log_dir, exist_ok=True)
    
    file_handler = logging.FileHandler(os.path.join(log_dir, "airhub.log"))
    file_handler.setFormatter(file_formatter)
    file_handler.setLevel(log_level)
    
    # Add handlers to logger
    logger.addHandler(console_handler)
    logger.addHandler(file_handler)
    logger.debug("Testing file handler for logging.")
    return logger