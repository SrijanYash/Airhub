"""
AirHub - Federated Learning Platform for Air Quality Prediction
Main FastAPI application entry point
"""
import uvicorn
from fastapi import FastAPI, Depends
from fastapi.middleware.cors import CORSMiddleware

from api.routes import predict_route, train_route
from utils.logger import setup_logger
import config

# Set up logger
logger = setup_logger(__name__)

# Create FastAPI app
app = FastAPI(
    title="AirHub API",
    description="Federated Learning Platform for Air Quality Prediction",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5500",
        "http://127.0.0.1:5500",
        "https://airhub-iota.vercel.app",
        "https://airhub.vercel.app"
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(predict_route.router, prefix="/api", tags=["predictions"])
app.include_router(train_route.router, prefix="/api", tags=["training"])

@app.get("/")
async def root():
    """Root endpoint that returns basic API information"""
    return {
        "message": "Welcome to AirHub API - Federated Learning Platform for Air Quality Prediction",
        "docs": "/docs",
        "endpoints": {
            "predict": "/api/predict",
            "train": "/api/train"
        }
    }

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy"}

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="AirHub API Server")
    parser.add_argument("--host", type=str, default=config.API_HOST, help="Host to bind to")
    parser.add_argument("--port", type=int, default=config.API_PORT, help="Port to bind to")
    parser.add_argument("--workers", type=int, default=config.API_WORKERS, help="Number of workers")
    parser.add_argument("--no-reload", action="store_true", help="Disable auto-reload")
    args = parser.parse_args()

    logger.info(f"Starting AirHub API on {args.host}:{args.port}")
    uvicorn.run(
        "main:app",
        host=args.host,
        port=args.port,
        workers=args.workers,
        reload=not args.no_reload
    )