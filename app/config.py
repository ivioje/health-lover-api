import json
from typing import List
from pydantic_settings import BaseSettings
from pydantic import Field


class Settings(BaseSettings):
    # Environment
    ENVIRONMENT: str = "development"
    DEBUG: bool = True

    # API
    API_V1_STR: str = "/api/v1"
    PROJECT_NAME: str = "My FastAPI App"
    PROJECT_VERSION: str = "1.0"

    # CORS
    BACKEND_CORS_ORIGINS: List[str] = Field(default_factory=list)

    # ML Models
    ML_MODELS_PATH: str = "./app/ml_models"

    # Logging
    LOG_LEVEL: str = "INFO"

    class Config:
        env_file = ".env"
        case_sensitive = True


settings = Settings()
