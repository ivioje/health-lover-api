import os
from typing import List
from pydantic import Field
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    # Environment
    ENVIRONMENT: str = Field(default_factory=lambda: os.getenv("ENVIRONMENT"))
    DEBUG: bool = Field(default_factory=lambda: os.getenv("DEBUG", "True").lower() == "true")
    
    # Database
    MONGODB_URL: str = Field(default_factory=lambda: os.getenv("MONGODB_URL"))
    DATABASE_NAME: str = Field(default_factory=lambda: os.getenv("DATABASE_NAME"))

    # API
    API_V1_STR: str = Field(default_factory=lambda: os.getenv("API_V1_STR"))
    PROJECT_NAME: str = Field(default_factory=lambda: os.getenv("PROJECT_NAME"))
    PROJECT_VERSION: str = 1.0

    # CORS
    BACKEND_CORS_ORIGINS: List[str] = Field(default_factory=lambda: os.getenv("BACKEND_CORS_ORIGINS".split(","))
    
    # ML Models
    ML_MODELS_PATH: str = Field(default_factory=lambda: os.getenv("ML_MODELS_PATH", "./app/ml_models"))
    
    # Logging
    LOG_LEVEL: str = Field(default_factory=lambda: os.getenv("LOG_LEVEL", "INFO"))
    
    class Config:
        env_file = ".env"
        case_sensitive = True


settings = Settings()
