"""
API routes package.
"""
from fastapi import APIRouter

from api.routes.predict_route import router as predict_router
from api.routes.train_route import router as train_router

# Create main router
router = APIRouter()

# Include routers
router.include_router(predict_router, tags=["predictions"])
router.include_router(train_router, tags=["training"])