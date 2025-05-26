from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
import logging
import sys
from pathlib import Path
import uvicorn


# Add the app directory to Python path
sys.path.append(str(Path(__file__).parent.parent))

from app.config import settings
from app.database import connect_to_mongo, close_mongo_connection
from app.services.ml_service import ml_service
from app.routers import recommendations

# Configure logging
logging.basicConfig(
    level=getattr(logging, settings.LOG_LEVEL),
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("app.log")
    ]
)

logger = logging.getLogger(__name__)

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager"""
    # Startup
    logger.info("Starting Health Lover Recommendation API...")
    
    try:
        await connect_to_mongo()
        logger.info("Database connection established")
        
        await ml_service.load_models()
        logger.info("ML models loaded")
        
        logger.info("Health Lover Recommendation API started successfully")
        
    except Exception as e:
        logger.error(f"Error during startup: {e}")
        raise
    
    yield
    
    # Shutdown
    logger.info("Shutting down Health Lover Recommendation API...")
    
    try:
        await close_mongo_connection()
        logger.info("Database connection closed")
        
        await ml_service.save_models()
        logger.info("ML models saved")
        
        logger.info("Health Lover Recommendation API shutdown complete")
        
    except Exception as e:
        logger.error(f"Error during shutdown: {e}")

app = FastAPI(
    title=settings.PROJECT_NAME,
    version=settings.PROJECT_VERSION,
    description="Machine Learning powered diet and recipe recommendation API",
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json",
    lifespan=lifespan
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.BACKEND_CORS_ORIGINS,
    allow_credentials=False,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)

# Include only recommendation router
app.include_router(recommendations.router, prefix=settings.API_V1_STR)

@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "Welcome to Health Lover Recommendation API",
        "version": settings.PROJECT_VERSION,
        "environment": settings.ENVIRONMENT,
        "docs": "/docs",
        "redoc": "/redoc",
        "purpose": "ML-powered diet and recipe recommendations"
    }

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    try:
        from app.database import db
        if db.client is None:
            raise Exception("Database not connected")
        
        # Ping database
        await db.client.admin.command('ping')
        
        # Check ML models
        models_status = {
            "models_loaded": ml_service.model_loaded,
            "content_model": ml_service.cosine_sim_matrix is not None,
            "collaborative_model": ml_service.collaborative_model is not None
        }
        
        return {
            "status": "healthy",
            "database": "connected",
            "ml_models": models_status,
            "environment": settings.ENVIRONMENT
        }
        
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        raise HTTPException(status_code=503, detail=f"Service unhealthy: {str(e)}")

@app.get("/api/v1/info")
async def api_info():
    """API information endpoint"""
    return {
        "name": settings.PROJECT_NAME,
        "version": settings.PROJECT_VERSION,
        "environment": settings.ENVIRONMENT,
        "purpose": "Machine Learning recommendation engine for diets and recipes",
        "features": [
            "Content-based diet recommendations",
            "Collaborative filtering recommendations", 
            "Hybrid recommendation system",
            "Recipe similarity analysis",
            "User interaction tracking"
        ],
        "endpoints": {
            "recommendations": f"{settings.API_V1_STR}/recommendations",
            "health": "/health",
            "info": f"{settings.API_V1_STR}/info"
        }
    }

# Error handlers
@app.exception_handler(404)
async def not_found_handler(request, exc):
    return {
        "error": "Not Found",
        "message": "The requested resource was not found",
        "status_code": 404
    }

@app.exception_handler(500)
async def internal_error_handler(request, exc):
    logger.error(f"Internal server error: {exc}")
    return {
        "error": "Internal Server Error",
        "message": "An unexpected error occurred",
        "status_code": 500
    }

if __name__ == "__main__":
    uvicorn.run(
        "app.main:app",
        host="0.0.0.0", 
        port=8000,
        reload=True
    )
