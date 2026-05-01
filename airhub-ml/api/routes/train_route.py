"""
Training endpoint for AirHub API.
"""
from fastapi import APIRouter, Depends, HTTPException, BackgroundTasks
from pydantic import BaseModel
from typing import Dict, Any, List, Optional
import os
import csv
from datetime import datetime

from federated.simulate_training import simulate_federated_learning
from utils.logger import setup_logger
import config

logger = setup_logger(__name__)

router = APIRouter()

class TrainingRequest(BaseModel):
    """Training request model."""
    num_clients: int = config.FL_MIN_CLIENTS
    num_rounds: int = config.FL_ROUNDS
    epochs_per_round: int = config.EPOCHS
    cities: Optional[List[Dict[str, str]]] = None  # List of {"city": city, "country": country}

class TrainingResponse(BaseModel):
    """Training response model."""
    status: str
    message: str
    training_config: Dict[str, Any]

# Global variable to track training status
training_status = {"is_training": False, "progress": 0, "message": ""}

def run_training_task(num_clients: int, num_rounds: int, epochs_per_round: int):
    """
    Run training task in background.
    
    Args:
        num_clients: Number of clients
        num_rounds: Number of rounds
        epochs_per_round: Number of epochs per round
    """
    global training_status
    
    try:
        training_status["is_training"] = True
        training_status["progress"] = 0
        training_status["message"] = "Starting training..."
        
        # Run federated learning simulation
        simulate_federated_learning(
            num_clients=num_clients,
            num_rounds=num_rounds,
            epochs_per_round=epochs_per_round
        )
        
        training_status["is_training"] = False
        training_status["progress"] = 100
        training_status["message"] = "Training completed successfully"
        
    except Exception as e:
        logger.error(f"Error in training: {str(e)}")
        training_status["is_training"] = False
        training_status["progress"] = 0
        training_status["message"] = f"Training failed: {str(e)}"

@router.post("/train", response_model=TrainingResponse)
async def train(request: TrainingRequest, background_tasks: BackgroundTasks):
    """
    Start federated learning training.
    
    Args:
        request: Training request
        background_tasks: Background tasks
        
    Returns:
        Training response
    """
    global training_status
    
    # Check if training is already in progress
    if training_status.get("is_training", False):
        return TrainingResponse(
            status="in_progress",
            message="Training is already in progress",
            training_config={
                "num_clients": request.num_clients,
                "num_rounds": request.num_rounds,
                "epochs_per_round": request.epochs_per_round
            }
        )
    
    try:
        logger.info(f"Training request: {request.dict()}")
        
        # Start training in background
        background_tasks.add_task(
            run_training_task,
            num_clients=request.num_clients,
            num_rounds=request.num_rounds,
            epochs_per_round=request.epochs_per_round
        )
        
        return TrainingResponse(
            status="started",
            message="Training started in background",
            training_config={
                "num_clients": request.num_clients,
                "num_rounds": request.num_rounds,
                "epochs_per_round": request.epochs_per_round
            }
        )
    
    except Exception as e:
        logger.error(f"Error starting training: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/train/status")
async def get_training_status():
    """
    Get training status.
    
    Returns:
        Training status
    """
    global training_status
    
    return training_status

@router.get("/flwr/metrics")
async def get_flower_metrics(limit: int = 50):
    """
    Return recent federated server evaluation metrics.
    
    Args:
        limit: Maximum number of latest records to return
        
    Returns:
        List of metrics rows with timestamp, loss, mae
    """
    records_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "federated", "records")
    metrics_path = os.path.join(records_dir, "metrics.csv")
    if not os.path.exists(metrics_path):
        return {"data": [], "message": "No metrics recorded yet"}
    data = []
    try:
        with open(metrics_path, "r", newline="") as f:
            reader = csv.DictReader(f)
            data = list(reader)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to read metrics: {e}")
    return {"data": data[-limit:]}

class AQIItem(BaseModel):
    date: str
    city: str
    country: str
    pm25: float
    pm10: float
    o3: float
    no2: float
    so2: float
    co: float

class WeatherItem(BaseModel):
    date: str
    city: str
    country: str
    temperature: float
    humidity: float
    pressure: float
    wind_speed: float
    wind_direction: float
    day_of_week: int
    month: int
    pm25: float
    pm10: float
    o3: float
    no2: float
    so2: float
    co: float

class IngestRequest(BaseModel):
    aqi: Optional[List[AQIItem]] = None
    weather: Optional[List[WeatherItem]] = None

@router.post("/data/ingest")
async def ingest_data(request: IngestRequest):
    try:
        datasets_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "datasets")
        if not os.path.exists(datasets_dir):
            datasets_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "data", "datasets")
        os.makedirs(datasets_dir, exist_ok=True)
        aqi_path = os.path.join(datasets_dir, "sample_aqi_data.csv")
        weather_path = os.path.join(datasets_dir, "sample_weather_data.csv")
        if request.aqi:
            with open(aqi_path, "w", newline="") as f:
                w = csv.writer(f)
                w.writerow(["date","city","country","pm25","pm10","o3","no2","so2","co"])
                for item in request.aqi:
                    w.writerow([item.date, item.city, item.country, item.pm25, item.pm10, item.o3, item.no2, item.so2, item.co])
        if request.weather:
            with open(weather_path, "w", newline="") as f:
                w = csv.writer(f)
                w.writerow(["date","city","country","temperature","humidity","pressure","wind_speed","wind_direction","day_of_week","month","pm25","pm10","o3","no2","so2","co"])
                for item in request.weather:
                    w.writerow([
                        item.date, item.city, item.country, item.temperature, item.humidity, item.pressure,
                        item.wind_speed, item.wind_direction, item.day_of_week, item.month,
                        item.pm25, item.pm10, item.o3, item.no2, item.so2, item.co
                    ])
        from data.preprocess import preprocess_data
        Xs, ys, X, Y = preprocess_data(save_to_file=True)
        return {
            "status": "ok",
            "saved": {
                "aqi": bool(request.aqi),
                "weather": bool(request.weather)
            },
            "shapes": {
                "X_scaled": list(Xs.shape),
                "y_scaled": list(ys.shape),
                "X": list(X.shape),
                "y": list(Y.shape)
            },
            "timestamp": datetime.utcnow().isoformat()
        }
    except Exception as e:
        logger.error(f"Failed to ingest data: {e}")
        raise HTTPException(status_code=500, detail=str(e))
