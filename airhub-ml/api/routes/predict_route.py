"""
Prediction endpoint for AirHub API.
"""
from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from typing import Dict, Any, Optional
import numpy as np

from model.predict import predict_tomorrow, get_past_predictions
from utils.logger import setup_logger
import config

logger = setup_logger(__name__)

router = APIRouter()

class PredictionRequest(BaseModel):
    """Prediction request model."""
    city: str = config.DEFAULT_CITY
    country: str = config.DEFAULT_COUNTRY

class PredictionResponse(BaseModel):
    """Prediction response model."""
    city: str
    country: str
    date: str
    aqi: float
    aqi_category: str
    temperature: float
    weather_type: str
    past_predictions: Optional[Dict[str, Any]] = None

@router.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest):
    """
    Predict tomorrow's air quality, temperature, and weather type.
    
    Args:
        request: Prediction request
        
    Returns:
        Prediction response
    """
    try:
        logger.info(f"Prediction request for {request.city}, {request.country}")
        
        # Get prediction
        prediction = predict_tomorrow(request.city, request.country)
        if not prediction:
            raise HTTPException(status_code=500, detail="Prediction failed")
        
        # Get past predictions
        past_df = get_past_predictions(request.city, request.country)
        past_serialized = None
        if past_df is not None and not past_df.empty:
            try:
                # Thoroughly clean any NaN values before serializing
                past_dict = past_df.replace({np.nan: None, np.inf: None, -np.inf: None}).to_dict(orient="records")
                past_serialized = {"rows": past_dict}
            except Exception as e:
                logger.warning(f"Failed to serialize past predictions: {e}")
                past_serialized = {"rows": []}
        
        # Coerce missing fields to sensible defaults
        temp = prediction.get("temperature")
        wt = prediction.get("weather_type")
        
        def safe_float(v):
            try:
                if v is None or np.isnan(v) or np.isinf(v):
                    return 0.0
                return float(v)
            except:
                return 0.0

        # Create response
        response = PredictionResponse(
            city=request.city,
            country=request.country,
            date=str(prediction.get("date", "")),
            aqi=safe_float(prediction.get("aqi")),
            aqi_category=str(prediction.get("aqi_category", "Unknown")),
            temperature=safe_float(temp),
            weather_type=str(wt) if wt else "Unknown",
            past_predictions=past_serialized
        )
        
        return response
    
    except HTTPException:
        # Re-raise HTTPExceptions as-is
        raise
    except Exception as e:
        logger.error(f"Error in prediction: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/predictions/history")
async def get_prediction_history():
    """
    Get past predictions.
    
    Returns:
        Past predictions
    """
    try:
        # Get past predictions
        past_df = get_past_predictions()
        return {"predictions": past_df.to_dict(orient="records")}
    
    except Exception as e:
        logger.error(f"Error getting prediction history: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
