"""
Shared FastAPI dependencies.
"""
from fastapi import Header, HTTPException, Depends
from typing import Optional

from utils.logger import setup_logger

logger = setup_logger(__name__)

async def get_api_key(x_api_key: Optional[str] = Header(None)):
    """
    Validate API key if configured.
    
    Args:
        x_api_key: API key from header
        
    Returns:
        API key if valid
        
    Raises:
        HTTPException: If API key is invalid
    """
    # For simplicity, we're not implementing actual API key validation
    # In a production environment, you would validate against a database or environment variable
    return x_api_key

async def get_token_header(x_token: str = Header(...)):
    """
    Validate token header.
    
    Args:
        x_token: Token from header
        
    Returns:
        Token if valid
        
    Raises:
        HTTPException: If token is invalid
    """
    # For simplicity, we're not implementing actual token validation
    # In a production environment, you would validate against a database or environment variable
    if x_token != "fake-super-secret-token":
        raise HTTPException(status_code=400, detail="X-Token header invalid")
    return x_token